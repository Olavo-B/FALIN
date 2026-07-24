# /*****************************************************************************/
#  * File: falin_cnn_quant.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-07-23
#  * License: [License Type]
#  * Description: Quantization-aware (QAT) version of FalinCNN using Brevitas.
#  * Mirrors the FalinCNN architecture layer-for-layer, replacing Conv2d/Linear
#  * with their quantized Brevitas equivalents. Intended as the pre-export
#  * model for the hls4ml/QONNX FPGA deployment flow (see Task 5 of
#  * misc/docs/fpga_implementation_plan.md).
# /*****************************************************************************/

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int32Bias


class FalinCNNQuant(nn.Module):
    def __init__(self, num_classes, in_channels=1, descriptor_size=64,
                 weight_bit_width=8, act_bit_width=8):
        super(FalinCNNQuant, self).__init__()

        arch_dim = [in_channels, 16, 32, descriptor_size]

        # Quantizes the raw GAF image once, at the network's entry point,
        # so every downstream Conv2d sees an already-quantized activation.
        self.quant_inp = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            bit_width=act_bit_width,
        )

        # Block 1: Initial feature extraction (Edges and transitions in STFT/GAF)
        self.layer1 = nn.Sequential(
            qnn.QuantConv2d(in_channels, arch_dim[1], kernel_size=3, stride=1, padding=1,
                             bias=False, weight_bit_width=weight_bit_width,
                             weight_quant=Int8WeightPerTensorFloat),
            nn.BatchNorm2d(arch_dim[1]),
            qnn.QuantReLU(bit_width=act_bit_width, act_quant=Int8ActPerTensorFloat),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Intermediate patterns
        self.layer2 = nn.Sequential(
            qnn.QuantConv2d(arch_dim[1], arch_dim[2], kernel_size=3, stride=1, padding=1,
                             bias=False, weight_bit_width=weight_bit_width,
                             weight_quant=Int8WeightPerTensorFloat),
            nn.BatchNorm2d(arch_dim[2]),
            qnn.QuantReLU(bit_width=act_bit_width, act_quant=Int8ActPerTensorFloat),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: Deep patterns and compression to form the feature descriptor
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(arch_dim[2], descriptor_size, kernel_size=3, stride=1, padding=1,
                             bias=False, weight_bit_width=weight_bit_width,
                             weight_quant=Int8WeightPerTensorFloat),
            nn.BatchNorm2d(descriptor_size),
            qnn.QuantReLU(bit_width=act_bit_width, act_quant=Int8ActPerTensorFloat),
            # AdaptiveAvgPool2d ensures the output is always (B, descriptor_size, 1, 1)
            # regardless of the input image size. Left unquantized (operates on the
            # already-quantized QuantReLU output) since Brevitas has no adaptive
            # variant; the classifier's QuantLinear re-quantizes on the way in.
            nn.AdaptiveAvgPool2d(1)
        )

        # Temporary Linear Classifier (Used ONLY if you want to train the CNN end-to-end before XGBoost)
        # Int32Bias needs a QuantTensor input carrying an input scale to derive
        # bias_scale = input_scale * weight_scale. Every other layer in this
        # network passes plain dequantized float tensors between modules (no
        # return_quant_tensor), so an explicit QuantIdentity(return_quant_tensor=True)
        # is inserted right before QuantLinear to (re)quantize the flattened
        # descriptor and hand QuantLinear the scale it needs. Without this, the
        # bias stays an unquantized float constant, which hls4ml's QONNX
        # frontend cannot fold into the final Add node (breaks scale
        # propagation - see Task 7 notes in misc/docs/fpga_implementation_plan.md).
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, bit_width=act_bit_width,
                               return_quant_tensor=True),
            qnn.QuantLinear(descriptor_size, num_classes,
                             bias=True, weight_bit_width=weight_bit_width,
                             weight_quant=Int8WeightPerTensorFloat,
                             bias_quant=Int32Bias)
        )

        self._initialize_weights()

    def forward(self, x, extract_features=False):
        """
        Args:
            x: Input images
            extract_features (bool): If True, returns the flattened vector for XGBoost.
        """
        x = self.quant_inp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # Output: (Batch, descriptor_size, 1, 1)

        # Flattens to (Batch, descriptor_size) -> E.g.: (16, 64)
        x = torch.flatten(x, 1)

        if extract_features:
            return x  # Returns directly to XGBoost!

        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # QuantConv2d/QuantLinear subclass nn.Conv2d/nn.Linear, so the same
        # isinstance checks used in FalinCNN apply unchanged here.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @classmethod
    def from_pretrained_falin_cnn(cls, falin_cnn, weight_bit_width=8, act_bit_width=8):
        """
        Builds a FalinCNNQuant and copies over Conv2d/Linear/BatchNorm2d weights
        from an already-trained FalinCNN checkpoint, so QAT fine-tuning starts
        from the float baseline instead of from scratch.
        """
        num_classes = falin_cnn.classifier[-1].out_features
        in_channels = falin_cnn.layer1[0].in_channels
        descriptor_size = falin_cnn.layer3[0].out_channels

        quant_model = cls(num_classes=num_classes, in_channels=in_channels,
                           descriptor_size=descriptor_size,
                           weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

        # strict=False: FalinCNNQuant's quant layers carry extra internal
        # parameters (e.g. layer*.*.act_quant...scaling_impl.value) that don't
        # exist in the float model and are left at their freshly-initialized
        # values. Conv2d/BatchNorm2d weight/bias names line up 1:1 with
        # FalinCNN since layer1/layer2/layer3 share the same Sequential
        # indices, so those are copied as-is.
        #
        # classifier is the one exception: FalinCNN's classifier is
        # Sequential(Dropout, Linear) -> Linear is index 1. FalinCNNQuant's
        # classifier is Sequential(Dropout, QuantIdentity, QuantLinear) ->
        # QuantLinear is index 2 (the QuantIdentity(return_quant_tensor=True)
        # was inserted so Int32Bias has an input scale to quantize the bias
        # against - see the comment above self.classifier). Remap the key
        # prefix explicitly instead of relying on name-for-name matching.
        state_dict = {
            (k.replace('classifier.1.', 'classifier.2.', 1) if k.startswith('classifier.1.') else k): v
            for k, v in falin_cnn.state_dict().items()
        }

        result = quant_model.load_state_dict(state_dict, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                "Unexpected keys when loading FalinCNN weights into FalinCNNQuant "
                f"(architecture mismatch): {result.unexpected_keys}"
            )

        return quant_model

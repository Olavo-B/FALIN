# /*****************************************************************************/
#  * File: mobilenet_v2.py
#  * Author: Olavo Alves Barros Silva
#  * Description: Custom implementation of MobileNetV2 architecture.
#  * Built with a modular design to easily extract intermediate 
#  * features for hybrid models (e.g., MobileNetV2 + XGBoost).
# /*****************************************************************************/

import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    """Standard Convolution -> BatchNorm -> ReLU6 pattern used throughout the network."""
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    """Main MobileNetV2 Block: Expansion -> Depthwise -> Pointwise (Linear)"""
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw (Pointwise) - Expansion
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        
        layers.extend([
            # dw (Depthwise)
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear (Pointwise without non-linear activation at the end)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, width_mult=1.0):
        """
        Args:
            num_classes (int): Number of classes for the final MLP (standard training).
            in_channels (int): Number of input image channels (e.g., 1 for GAF, 3 for RGB).
            width_mult (float): Width multiplier to adjust the network size.
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        # Official MobileNetV2 setting: t (expansion), c (channels), n (repeats), s (stride)
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building the first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        
        features = [ConvBNReLU(in_channels, input_channel, stride=2)]
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                
        # Building the last convolutional layer before pooling
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        
        # Grouping all features
        self.features = nn.Sequential(*features)

        # Final classifier (MLP)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # Weight initialization
        self._initialize_weights()

    def forward(self, x, extract_features=False):
        """
        Forward pass of the network.
        
        Args:
            x (Tensor): Batch of input images.
            extract_features (bool): If True, returns the Global Average Pooling vector 
                                     (skips the MLP). Ideal for feeding into XGBoost.
        """
        # Extract spatial feature maps (B, C, H, W)
        x = self.features(x)
        
        # Global Average Pooling (B, C, H, W) -> (B, C)
        x = x.mean([2, 3])
        
        if extract_features:
            return x
            
        # Pass through the classifier (only for standard CNN training)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

# /*****************************************************************************/
#  * File: falin_cnn.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-06-19
#  * License: [License Type]
#  * Description: Super lightweight 2D CNN architecture (Custom TinyML) 
#  * optimized for FPGA. Extracts features from GAF/STFT images for the classifier.
# /*****************************************************************************/

import torch
import torch.nn as nn

class FalinCNN(nn.Module):
    def __init__(self, num_classes, in_channels=1, descriptor_size=64):
        super(FalinCNN, self).__init__()
        
        # 2026-06-20 21:45:27 - Olavo: Adjusted the architecture dimensions to further reduce complexity for FPGA deployment.
        arch_dim = [in_channels, 16, 32, descriptor_size]
        
        # Block 1: Initial feature extraction (Edges and transitions in STFT/GAF)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, arch_dim[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(arch_dim[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Intermediate patterns
        self.layer2 = nn.Sequential(
            nn.Conv2d(arch_dim[1], arch_dim[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(arch_dim[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Deep patterns and compression to form the feature descriptor
        self.layer3 = nn.Sequential(
            nn.Conv2d(arch_dim[2], descriptor_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(descriptor_size),
            nn.ReLU(inplace=True),
            # AdaptiveAvgPool2d ensures the output is always (B, descriptor_size, 1, 1)
            # regardless of the input image size!
            nn.AdaptiveAvgPool2d(1) 
        )

        # Temporary Linear Classifier (Used ONLY if you want to train the CNN end-to-end before XGBoost)
        self.classifier = nn.Sequential(
            # 2026-06-20 22:10:15 - Olavo: Changed the dropout rate to 0.5 to further prevent overfitting, especially since the CNN is very lightweight.
            nn.Dropout(0.5), # Dropout before the final layer to prevent overfitting
            nn.Linear(descriptor_size, num_classes)
        )

        self._initialize_weights()

    def forward(self, x, extract_features=False):
        """
        Args:
            x: Input images
            extract_features (bool): If True, returns the flattened vector for XGBoost.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # Output: (Batch, descriptor_size, 1, 1)
        
        # Flattens to (Batch, descriptor_size) -> E.g.: (16, 64)
        x = torch.flatten(x, 1) 
        
        if extract_features:
            return x # Returns directly to XGBoost!
            
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
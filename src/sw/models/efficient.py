# /*****************************************************************************/
#  * File: efficientnet.py
#  * Author: Olavo Alves Barros Silva
#  * Description: Custom implementation of a lightweight EfficientNet (B0-like).
#  * Includes Squeeze-and-Excitation (SE) blocks and Swish activation.
#  * Designed for 1D-to-2D temporal data classification (e.g., GAF/STFT).
# /*****************************************************************************/

import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Squeeze
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(), # Swish activation
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid() # Excitation
        )

    def forward(self, x):
        return x * self.se(x)

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution with SE block."""
    def __init__(self, inp, oup, stride, expand_ratio, se_ratio=0.25):
        super(MBConv, self).__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # 1. Expansion Phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ])
            
        # 2. Depthwise Convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ])
        
        # 3. Squeeze-and-Excitation
        reduced_dim = max(1, int(inp * se_ratio))
        layers.append(SqueezeExcitation(hidden_dim, reduced_dim))
        
        # 4. Pointwise (Linear) Convolution
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, width_mult=1.0):
        super(EfficientNetCustom, self).__init__()
        
        # Configuration: [expand_ratio, channels, repeats, stride]
        mbconv_settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 40, 2, 2],
            [6, 80, 3, 2],
            [6, 112, 3, 1],
            [6, 192, 4, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult)
        
        # Stem
        features = [
            nn.Conv2d(in_channels, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.SiLU()
        ]
        
        # Blocks
        for t, c, n, s in mbconv_settings:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(MBConv(input_channel, output_channel, stride, t))
                input_channel = output_channel
                
        # Head
        features.extend([
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.SiLU()
        ])
        
        self.features = nn.Sequential(*features)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x, extract_features=False):
        x = self.features(x)
        x = x.mean([2, 3]) # Global Average Pooling
        
        if extract_features:
            return x # [Batch, 1280] -> Feed to XGBoost
            
        x = self.classifier(x)
        return x
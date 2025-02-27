import torch
import torch.nn as nn
from math import ceil 

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3], 
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 5],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

phi_values = {
    # tuple: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)
}

class CNNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride,
                 padding,
                 groups=1
                 ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            groups=groups,
                            bias=False # since we are using batch norm
                            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))
    

# calculate attention scores for each channel
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            # (batch_size, channel x height x width) -> (batch_size, channel x 1 x 1)
            nn.AdaptiveAvgPool2d(1),

            # (batch_size, channel x 1 x 1) -> (batch_size, reduced_dim x 1 x 1)
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),

            # (batch_size, reduced_dim x 1 x 1) -> (batch_size, channel x 1 x 1)
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (batch_size, channel x height x width) * (batch_size, channel x 1 x 1) -> element wise multiplication
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 expand_ratio,
                 reduction=4, #squeeze excitation
                 survival_proba=0.8 #stochastic depth 
                 ):
        super(InvertedResidualBlock,self).__init__()
        self.survival_proba = survival_proba
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels,hidden_dim, kernel_size=3, stride=1, padding=1
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    # this is a bit like light dropout but not during testing 
    def stochastic_depth(self, x):
        if not self.training:
            return x 

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_proba
        return torch.div(x, self.survival_proba) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self,
                 in_channels,
                 version,
                 num_classes
                 ):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(in_channels,width_factor, depth_factor, last_channels)
        self.classifer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, dropout_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, dropout_rate
    
    def create_features(self, in_channels, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(in_channels, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ration, channels, repeats, stride, kernel_size in base_model:
            # some math trick for squeeze excitation
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeat = ceil(repeats * depth_factor)
            for layer in range(layers_repeat):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride if layer == 0 else 1,
                        padding=kernel_size // 2,
                        expand_ratio=expand_ration
                    )
                )

                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifer(x.view(x.shape[0],-1))

class EfficientNet_B0(EfficientNet):
    def __init__(self,
                 in_channels,
                 num_classes
                 ):
        version = "b0"
        super(EfficientNet_B0, self).__init__(version=version, in_channels=in_channels, num_classes=num_classes)

class EfficientNet_B1(EfficientNet):
    def __init__(self,
                 in_channels,
                 num_classes
                 ):
        version = "b1"
        super(EfficientNet_B1, self).__init__(version=version, in_channels=in_channels, num_classes=num_classes)

class EfficientNet_B2(EfficientNet):
    def __init__(self,
                 in_channels,
                 num_classes
                 ):
        version = "b2"
        super(EfficientNet_B2, self).__init__(version=version, in_channels=in_channels, num_classes=num_classes)

class EfficientNet_B3(EfficientNet):
    def __init__(self,
                 in_channels,
                 num_classes
                 ):
        version = "b3"
        super(EfficientNet_B3, self).__init__(version=version, in_channels=in_channels, num_classes=num_classes)

class EfficientNet_B4(EfficientNet):
    def __init__(self,
                 in_channels,
                 num_classes
                 ):
        version = "b4"
        super(EfficientNet_B4, self).__init__(version=version, in_channels=in_channels, num_classes=num_classes)

class EfficientNet_B5(EfficientNet):
    def __init__(self,
                 in_channels,
                 num_classes
                 ):
        version = "b5"
        super(EfficientNet_B5, self).__init__(version=version, in_channels=in_channels, num_classes=num_classes)

class EfficientNet_B6(EfficientNet):
    def __init__(self,
                 in_channels,
                 num_classes
                 ):
        version = "b6"
        super(EfficientNet_B6, self).__init__(version=version, in_channels=in_channels, num_classes=num_classes)
        
class EfficientNet_B7(EfficientNet):
    def __init__(self,
                 in_channels,
                 num_classes
                 ):
        version = "b7"
        super(EfficientNet_B7, self).__init__(version=version, in_channels=in_channels, num_classes=num_classes)
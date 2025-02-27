import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    """A basic convolution block: Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DynamicMBConv(nn.Module):
    """
    A Mobile Inverted Bottleneck Convolution (MBConv) block that supports dynamic
    configurations. In a full OFA implementation, candidates for kernel size, expansion ratio,
    and other hyperparameters can be selected from a predefined search space.
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio, kernel_size):
        super(DynamicMBConv, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        
        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, stride=1))
        # Depthwise convolution
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                 stride=stride, groups=hidden_dim))
        # Pointwise linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class OFANet(nn.Module):
    """
    A simplified Once-for-All (OFA) network architecture.
    
    The network consists of:
      - A stem convolution
      - A sequence of MBConv blocks with varying configurations
      - A final head with a 1x1 convolution, global average pooling, and a classifier.
      
    The 'arch_settings' list defines the configuration for each stage in the network:
      (expansion_ratio, output_channels, num_repeats, stride, kernel_size)
    """
    def __init__(self, num_classes=1000):
        super(OFANet, self).__init__()
        # Stem: initial convolution
        self.stem = ConvBNReLU(3, 16, kernel_size=3, stride=2)
        
        # Define architecture settings.
        # Each tuple: (expand_ratio, out_channels, num_repeats, stride, kernel_size)
        self.arch_settings = [
            (1,   16, 1, 1, 3),
            (6,   24, 2, 2, 3),
            (6,   40, 2, 2, 5),
            (6,   80, 3, 2, 3),
            (6,  112, 3, 1, 3),
            (6,  192, 4, 2, 5),
            (6,  320, 1, 1, 3),
        ]
        
        input_channel = 16
        self.blocks = nn.ModuleList()
        # Build MBConv blocks according to the settings.
        for expand_ratio, output_channel, num_repeats, stride, kernel_size in self.arch_settings:
            for i in range(num_repeats):
                s = stride if i == 0 else 1
                self.blocks.append(DynamicMBConv(input_channel, output_channel, s,
                                                   expand_ratio, kernel_size))
                input_channel = output_channel
                
        # Final layers
        self.head = ConvBNReLU(input_channel, 1280, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn

"""
3x3 Depthwise Conv -> BN -> ReLU -> 1x1 Conv -> BN -> ReLU

Depthwise convolutions are controlled by the groups parameter:
    groups=1 (default) -> a single filter applies across all input channels
    groups=in_channels -> each input channel has its own separater filter
"""
class DepthWiseSeparableConv(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 stride=1
                 ):
        super(DepthWiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) #inplace modifies the tensor directly, instead of creating a new copy

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x 


class MobileNetV1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(MobileNetV1, self).__init__()
        def conv_bn_relu(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.model = nn.Sequential(
            conv_bn_relu(in_channels, 32, stride=2),
            DepthWiseSeparableConv(32, 64, stride=1),
            DepthWiseSeparableConv(64, 128, stride=2),
            DepthWiseSeparableConv(128, 128, stride=1),
            DepthWiseSeparableConv(128, 256, stride=2),
            DepthWiseSeparableConv(256, 256, stride=1),
            DepthWiseSeparableConv(256, 512, stride=2),
            *[DepthWiseSeparableConv(512, 512, stride=1) for _ in range(5)],
            DepthWiseSeparableConv(512, 1024, stride=2),
            DepthWiseSeparableConv(1024, 1024, stride=1), # outputs (batch_size, 1024, 7, 7)
            nn.AdaptiveAvgPool2d(1), #pools 7 x 7 down to 1 x 1 -> outputs (batch_size, 1024, 1, 1)
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.model(x)


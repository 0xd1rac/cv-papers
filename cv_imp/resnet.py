import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1  # This should be a class attribute

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    """
    The Bottleneck block uses an expansion factor of 4, which means the number of output
    channels after the third convolution is four times the number of output channels after
    the first convolution.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 convolution to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution to process spatial dimensions
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution to restore channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetBasicBlock(nn.Module):
    def __init__(self, layers, input_channels=3, num_classes=1000):
        super(ResNetBasicBlock, self).__init__()

        # Initial convolutional layer with kernel size 7x7, stride 2, and padding 3
        # This layer increases the number of channels from input_channels (usually 3 for RGB images) to 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Track the number of input channels for subsequent layers
        self.in_channels = 64

        # Batch normalization and ReLU activation for the initial convolutional layer
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling to reduce the spatial dimensions of the feature maps
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # First residual block layer - no downsampling
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])

        # Second residual block layer - downsampling occurs due to stride=2
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)

        # Third residual block layer - downsampling occurs due to stride=2
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)

        # Fourth residual block layer - downsampling occurs due to stride=2
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        # Adaptive average pooling to reduce each feature map to a 1x1 spatial dimension
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to map the feature maps to the number of classes
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        # Create a residual block layer

        downsample = None

        # If stride != 1 or input channels don't match output channels (after expansion), use downsampling
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []

        # Add the first block to the layer, with downsampling if needed
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        # Update in_channels for the subsequent blocks
        self.in_channels = out_channels * block.expansion

        # Add the remaining blocks (if any) to the layer
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        # Return the layer as a sequential container
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the initial convolutional layer, batch normalization, and ReLU activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Forward pass through the max pooling layer
        x = self.maxpool(x)

        # Forward pass through each of the residual block layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Forward pass through the adaptive average pooling layer
        x = self.avgpool(x)

        # Flatten the output from the pooling layer to prepare it for the fully connected layer
        x = torch.flatten(x, 1)

        # Forward pass through the fully connected layer to get the final output
        x = self.fc(x)

        return x
    
class ResNetBottleneck(nn.Module):
    def __init__(self, layers, input_channels, num_classes=1000):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.in_channels = 64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class Resnet18(ResNetBasicBlock):
    def __init__(self, 
                 input_channels: int, 
                 num_classes:int = 1000
                 ):
        layers = [2,2,2,2]
        super(Resnet18, self).__init__(layers, 
                                       input_channels=input_channels,
                                       num_classes=num_classes
                                       )

class ResNet34(ResNetBasicBlock):
    def __init__(self, 
                 input_channels: int, 
                 num_classes:int = 1000
                 ):
        layers = [3, 4, 6, 3]
        super(ResNet34, self).__init__(layers, 
                                       input_channels=input_channels,
                                       num_classes=num_classes
                                       )


class ResNet50(ResNetBottleneck):
    def __init__(self, 
                 input_channels: int, 
                 num_classes:int = 1000
                 ):
        layers = [3, 4, 6, 3]
        super(ResNet50, self).__init__(layers, 
                                       input_channels=input_channels,
                                       num_classes=num_classes
                                       )

class ResNet101(ResNetBottleneck):
    def __init__(self, 
                 input_channels: int, 
                 num_classes:int = 1000
                 ):
        layers = [3, 4, 23, 3]
        super(ResNet101, self).__init__(layers, 
                                        input_channels=input_channels,
                                        num_classes=num_classes
                                        )
class ResNet152(ResNetBottleneck):
    def __init__(self, 
                 input_channels: int, 
                 num_classes:int = 1000
                 ):
        layers = [3, 8, 36, 3]
        super(ResNet152, self).__init__(layers, 
                                        input_channels=input_channels,
                                        num_classes=num_classes
                                        )

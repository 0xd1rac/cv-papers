import torch
import torch.nn as nn

conv_arch_config = [
    # (kernel size, output_channels, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",

    # List: tuples and last int represents number of repeats
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",

    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 **kwargs
                 ):
        super(CNNBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn_layer = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.bn_layer(self.conv_layer(x)))

class Yolo_v1(nn.Module):
    def __init__(self,
                 in_channels=3,
                 **kwargs
                 ):
        super(Yolo_v1, self).__init__()
        self.conv_arch_config = conv_arch_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.conv_arch_config)
        self.fcs = self._create_fcs(**kwargs)

    def _create_conv_layers(self,conv_arch_config):
        layers = []
        in_channels = self.in_channels
        for arch in conv_arch_config:
            if isinstance(arch, tuple):
                kernel_size, out_channels, stride, padding = arch
                layers.append(
                    CNNBlock(
                        in_channels, 
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                )
                in_channels = out_channels  # Update in_channels after each CNNBlock
                
            elif isinstance(arch, str):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif isinstance(arch, list):
                conv_1, conv_2, num_repeats = arch
                for _ in range(num_repeats):
                    layers.append(
                        CNNBlock(
                            in_channels,
                            conv_1[1],
                            kernel_size=conv_1[0],
                            stride=conv_1[2],
                            padding=conv_1[3]
                        )
                    )
                    layers.append(
                         CNNBlock(
                            conv_1[1],
                            conv_2[1],
                            kernel_size=conv_2[0],
                            stride=conv_2[2],
                            padding=conv_2[3]
                         )
                    )
                    in_channels = conv_2[1]  # Update in_channels after each pair of CNNBlocks

        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Linear(1024 * 4 * 4 , 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5))
        )
        
    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

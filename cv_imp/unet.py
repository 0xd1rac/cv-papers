import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1, #binary image segmentation
                 features=[64,128,256,512]
                 ):
        super(UNet, self).__init__()
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down sample
        for feature in features:
            self.downsample.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # upsample 
        for feature in reversed(features):
            self.upsample.append(nn.ConvTranspose2d(feature*2, feature,kernel_size=2, stride=2))
            self.upsample.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downsample:
            x = down(x) 
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.upsample), 2):
            x = self.upsample[idx](x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = F.resize(x, size=skip.shape[2:])

            concat_skip = torch.cat((skip, x), dim=1)
            x = self.upsample[idx+1](concat_skip)

        return self.final_conv(x)
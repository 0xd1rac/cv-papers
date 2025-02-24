import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union
from torch import Tensor

# class AttentionPooling(nn.Module):
# Need to study bottlneck layer again 
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
                 in_channels:int,
                 bottleneck_channels:int,
                 stride:int=1
                 ):
        """
        Args:
            in_channels (int): Number of input channels.
            bottleneck_channels (int): Number of channels in the intermediate (bottleneck) layers.
            stride (int): Stride for spatial downsampling. If greater than 1, downsampling is applied.
        """
        super().__init__()

        # 1x1 convolution to reduce channels 
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 3x3 convolution (with padding to preservee spatial size)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # If stride > 1, apply average pooling after the second convolution (downsampling - reduces the resolution)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(kernel_size=stride)
        else:
            self.avgpool = nn.Identity()
        
        #1x1 convolution to expand the channels
        out_channels = bottleneck_channels * self.expansion
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        # Make sure the identity branch is the same shape as the output of the main put
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(OrderedDict([
                ("avgpool", nn.AvgPool2d(kernel_size=stride)), # downsamples the resolution
                ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)), # matchse the channels
                ("bn", nn.BatchNorm2d(out_channels))
            ]))
    
    def forward(self, x:Tensor)->Tensor:
        identity = x 

        # 1x1 conv
        out = self.relu1(self.bn1(self.conv1(x)))

        # 3x3 conv
        out = self.relu2(self.bn2(self.conv2(2)))

        # Apply average pooling if stride > 1
        out = self.avgpool(out)

        # final 1x1 conv to expand channels 
        out = self.bn3(self.conv3(out))

        # Downsample the identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu3(out + identity)
        return out 
        
class AttentionPool2d(nn.Module):
    def __init__(self, 
                 spacial_dim:int,
                 embed_dim: int,
                 num_head: int,
                 output_dim: int=None         
                 ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.




class ModifiedResNet(nn.Module):
    """
    modified ResNet wtih 
        - 3 stem convolutions (initial conv of a nn) with avg pool instead of max pool
        - Anti-aliasing strided convolutions -> uses avg pooling before strided convolutions 
        - Final pooling layer as QKV attention -> Replace Global Average Pooling with self-attention
    
    Args:
        blocks_per_stage (list or tuple of int): Number of Bottleneck blocks in each stage.
        final_output_dim (int): The desired output dimension after the attention pooling.
        num_attention_heads (int): The number of attention heads for the final pooling.
        input_image_size (int): The height/width of the input image (assumed square). Default is 224.
        base_channels (int): The base number of channels in the network. Default is 64.
    """
    def __init__(self, 
                 blocks_per_stage:int,
                 final_output_dim:int,
                 num_attention_heads:int,
                 input_image_size:int=224, 
                 base_channels:int=64
                 ):
        
        super().__init__()
        self.final_output_dim = final_output_dim
        self.input_image_size = input_image_size

        # --- Stem: 3-layer convolutional stem ---        
        # First stem layer: reduce spatial resolution and increase channels from 3 to base_channels//2.
        self.conv1 = nn.Conv2d(3, base_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)


        # Second stem layer: keep channel count constant 
        self.conv2 = nn.Conv2d(base_channels // 2, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)

        # Third stem layer: increase channels to base channels 
        self.conv3 = nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels)
        self.relu3 = nn.ReLU(inplace=True)

        # Downsample spatially by a factor of 2 using pooling 
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

        # --- Residual Layers (stages) ---
        # This mutable variable keeps track of the number of channels from the previous stage.
        self.current_channels = base_channels

        # Each stage is created by _make_stage.
        self.stage1 = self._make_stage(target_channels=base_channels, num_blocks=blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(target_channels=base_channels * 2, num_blocks=blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(target_channels=base_channels * 4, num_blocks=blocks_per_stage[2], stride=2)
        self.stage4 = self._make_stage(target_channels=base_channels * 8, num_blocks=blocks_per_stage[3], stride=2)

        # Final feature embedding dimension is typically base_channels * 32 for ResNets.
        final_feature_dim = base_channels * 32

        # Final pooling layer uses attention-base pooling
        self.attention_pool = AttentionPool2d(
            input_size=input_image_size // 32,
            embed_dim=final_feature_dim,
            num_heads=num_attention_heads,
            output_dim=final_output_dim
        )

    def _make_stage(self, target_channels:int, num_blocks: int, stride:int = 1) -> nn.Sequential:
        """
        Creates one stage (sequence) of Bottleneck blocks.
        
        Args:
            target_channels (int): The number of channels (before expansion) for this stage.
            num_blocks (int): How many Bottleneck blocks in this stage.
            stride (int): The stride to use in the first block of the stage.
        
        Returns:
            nn.Sequential: The sequential stage of Bottleneck blocks.
        """
        """
        In a bottleneck block the internal convolution work on a reduced channel dimension (this is what 
        target_channels sets) and then, at the end of the block the output channels are increased by 
        a fixed expansion factor (for example, 4)
        
        """
        # The first block may downsample spatially (if stride > 1) or adjust channel count.
        blocks = [Bottleneck(self.current_channels, target_channels, stride)]

        # Update the current channel count after expansion
        self.current_channels = target_channels * Bottleneck.expansion

        for _ in range(1, num_blocks):
            blocks.append(Bottleneck(self.current_channels, target_channels))
        
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:

        # -- Stem Forward Pass --
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avg_pool(x)

        # --- Residual Stages ---
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # --- Final Attention Pooling ---
        x = self.attention_pool(x)
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

"""
Transformer encoder block
"""
class ResidualAttentionBlock(nn.Module):
    def __init__(self, 
                 d_model:int,
                 n_head:int,
                 attn_mask:torch.Tensor=None
                 ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict(
            [
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ]
        ))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
    
    def attention(self, x:torch.Tensor):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

class Transformer(nn.Module):
    def __init__(self, d_model:int, layers:int, heads:int, attn_mask:torch.Tensor=None):
        super().__init__()
        self.d_model = d_model
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(d_model, heads, attn_mask) for _ in range(layers)]    
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
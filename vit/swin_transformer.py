import torch 
import torch.nn as nn
from timm.models.layers import to_2tuple
class PatchEmbed(nn.Module):
    def  __init__(self,
                  img_size=224,
                  patch_size=4,
                  in_chans=3,
                  embed_dim=96,
                  norm_layer=None
                  ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans 
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans, 
                              out_channels=embed_dim, 
                              kernel_size=patch_size,
                              stride=patch_size
                              )
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) does not match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # (B, patch_height * patch_width, C)
        x = self.proj(x).flatten(2).transpose(1,2) 

        if self.norm is not None:
            x = self.norm(x)
        
        return x
    
class WindowAttention(nn.Module):

    

import torch
import torch.nn as nn
import numpy as np 
import torchvision.transforms as transforms
from timm.models.vision_transformer import VisionTransformer

class PatchEmbedding(nn.Module):
    """ Splits the image into patches and embeds them with a linear projection. """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2 
    
        # Linear projection for patch embeddings
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.rand(1, self.num_patches, embed_dim))
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        assert height == width == self.img_size, f"Image size must match the model, ({self.img_size}x{self.img_size})"

        # Extract patches and create a patch vector of size = C x patch_size^2
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, num_channels * self.patch_size * self.patch_size) # flatten each patch into a row vector (batch_size, num_patches, num_channels * patch_size * patch_size)

        # Project patches into embeddings
        x = self.proj(x) + self.pos_embed
        return x 
    
def mask_patches(patch_embeddings, mask_ratio=0.75):
    """Randomly masks patches and returns visible patches and mask indices."""
    batch_size, num_patches, embed_dim = patch_embeddings.shape
    num_masked = int(mask_ratio * num_patches)

    # Generate random mask indices 
    indices = np.argsort(np.random.rand(batch_size, num_patches), axis=1)
    masked_indices = indices[:, :num_masked]
    visible_indices = indices[:, num_masked:] 
    
    # Get visible patches 
    visible_indices = torch.tensor(visible_indices, dtype=torch.long, device=patch_embeddings.device)
    visible_patches = torch.gather(
        patch_embeddings, 1, 
        visible_indices.unsqueeze(-1).expand(-1, -1, patch_embeddings.shape[-1])
    )
    return visible_patches, visible_indices, masked_indices


class MAEEncoder(nn.Module):
    """Encoder for Masked Autoencoder (MAE). Uses a ViT."""
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=3, embed_dim=embed_dim)
        self.encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, num_classes=0, mlp_ratio=4.0, qkv_bias=True
        )

    def forward(self, x, mask_ratio=0.75):
        print(f"This is MAEncoder forward function: {x.shape}")
        patch_embeddings = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        visible_patches, visible_indices, masked_indices = mask_patches(patch_embeddings, mask_ratio)

        # FIX: Use forward_features() instead of forward()
        encoded_features = self.encoder.forward_features(visible_patches)  

        return encoded_features, visible_indices, masked_indices

class MAEDecoder(nn.Module):
    """MAE Decoder that reconstructs masked image patches"""
    def __init__(self, embed_dim=768, decoder_embed_dim=512, depth=8, num_heads=8, patch_size=16, img_size=224):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=decoder_embed_dim,
            depth=depth, num_heads=num_heads, num_classes=0, mlp_ratio=4.0, qkv_bias=True
        )
        self.reconstruction_head = nn.Linear(decoder_embed_dim, patch_size * patch_size * 3)

    def forward(self, encoded_features, visible_indices, masked_indices):
        batch_size, num_visible_patches, _ = encoded_features.shape
        num_total_patches = (self.img_size // self.patch_size) ** 2

        full_tokens = torch.zeros(batch_size, num_total_patches, self.decoder_embed_dim, device=encoded_features.device)
        encoded_features = self.proj(encoded_features)

        full_tokens.scatter_(
            1, visible_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim), encoded_features
        )

        mask_tokens = self.mask_token.expand(batch_size, masked_indices.shape[1], -1)
        full_tokens.scatter_(
            1, masked_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim), mask_tokens
        )

        # FIX: Use forward_features() instead of forward()
        decoder_features = self.decoder.forward_features(full_tokens)  

        reconstructed_patches = self.reconstruction_head(decoder_features)
        return reconstructed_patches

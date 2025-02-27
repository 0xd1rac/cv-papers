import torch
import torch.nn as nn
import numpy as np 
from typing import Tuple
import torchvision.transforms as transforms
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor


def extract_patches(x: Tensor, patch_size: int) -> Tensor:
    """
    Extracts flattened patches from the original image (ground truth for reconstruction).

    Args:
        x (Tensor): Input image tensor of shape (B, C, H, W), where H = W.
        patch_size (int): The size of each patch (assumed to be square).

    Returns:
        Tensor: A tensor of shape (B, num_patches, patch_dim), where patch_dim is equal to C * patch_size * patch_size.
    """
    batch_size, channels, _, _ = x.shape
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        batch_size, -1, channels * patch_size * patch_size
    )
    return patches

class PatchEmbedding(nn.Module):
    """ Splits the image into patches and embeds them with a linear projection. """
    def __init__(self, 
                 img_size:int=224, 
                 patch_size:int=16, 
                 in_channels:int=3, 
                 embed_dim:int=768
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2 
    
        # Linear projection for patch embeddings
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.rand(1, self.num_patches, embed_dim))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape
        assert height == width == self.img_size, f"Image size must match the model, ({self.img_size}x{self.img_size})"

        # Extract patches and create a patch vector of size = C x patch_size^2
        x = extract_patches(x, self.patch_size)

        # Project patches into embeddings
        x = self.proj(x) + self.pos_embed
        return x 
    
def mask_patches(patch_embeddings: Tensor, mask_ratio: float = 0.75) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Randomly masks patches and returns visible patches and mask indices.

    Args:
        patch_embeddings (torch.Tensor): Input tensor of shape (B, num_patches, embed_dim).
        mask_ratio (float, optional): Fraction of patches to mask. Default is 0.75.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - visible_patches (torch.Tensor): The visible patches after masking, of shape (B, num_visible, embed_dim).
            - visible_indices (torch.Tensor): The indices of the visible patches, of shape (B, num_visible).
            - masked_indices (torch.Tensor): The indices of the masked patches, of shape (B, num_masked).
    """
    batch_size, num_patches, embed_dim = patch_embeddings.shape
    num_masked = int(mask_ratio * num_patches)

    # Generate random mask indices 
    indices = np.argsort(np.random.rand(batch_size, num_patches), axis=1)
    masked_indices = indices[:, :num_masked]
    visible_indices = indices[:, num_masked:] 
    
   # Convert indices to torch tensors 
    visible_indices = torch.tensor(visible_indices, dtype=torch.long, device=patch_embeddings.device)
    masked_indices = torch.tensor(masked_indices, dtype=torch.long, device=patch_embeddings.device)

    # Gather the visible patches based on visible_indices
    visible_patches = torch.gather(
        patch_embeddings, 1, 
        visible_indices.unsqueeze(-1).expand(-1, -1, patch_embeddings.shape[-1])
    )
    return visible_patches, visible_indices, masked_indices


class MAEEncoder(nn.Module):
    """
    Encoder for Masked Autoencoder (MAE) using a Vision Transformer (ViT).

    This module performs the following steps:
      1. Extracts patches from the input image via a PatchEmbedding module.
      2. Randomly masks a fraction of these patches.
      3. Encodes the visible patches using a ViT (bypassing the internal patch embedding).
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768, 
                 depth: int = 12, num_heads: int = 12) -> None:
        """
        Args:
            img_size (int): The height/width of the input image (assumed square).
            patch_size (int): The size of each patch (assumed square).
            embed_dim (int): The dimension of the patch embeddings.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads in each transformer block.
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, 
                                          in_channels=3, embed_dim=embed_dim)
        
        # Initialize a Vision Transformer without a classification head (num_classes=0)
        self.encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, num_classes=0, mlp_ratio=4.0, qkv_bias=True
        )

    def forward(self, x: Tensor, mask_ratio: float = 0.75) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for the MAE encoder.

        Args:
            x (Tensor): Input image tensor of shape (B, 3, img_size, img_size).
            mask_ratio (float, optional): Fraction of patches to mask. Default is 0.75.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - encoded_features (Tensor): Encoded features of the visible patches with shape (B, num_visible, embed_dim).
                - visible_indices (Tensor): Indices of the visible patches with shape (B, num_visible).
                - masked_indices (Tensor): Indices of the masked patches with shape (B, num_masked).
        """
        patch_embeddings: Tensor = self.patch_embed(x)  # Shape: (B, num_patches, embed_dim)
        
        visible_patches, visible_indices, masked_indices = mask_patches(patch_embeddings, mask_ratio)

        # Directly process the visible patches through the transformer blocks and normalization
        encoded_features: Tensor = self.encoder.blocks(visible_patches)
        encoded_features = self.encoder.norm(encoded_features)

        return encoded_features, visible_indices, masked_indices

class MAEDecoder(nn.Module):
    """
    MAE Decoder that reconstructs masked image patches.

    This module projects the encoder output into the decoder embedding dimension,
    creates a full token sequence (populated with encoded visible tokens and learnable
    mask tokens), and then processes these tokens with a Vision Transformer to
    reconstruct the original patches.
    
    """
    def __init__(self, 
                 embed_dim: int = 768, 
                 decoder_embed_dim: int = 512, 
                 depth: int = 8, 
                 num_heads: int = 8, 
                 patch_size: int = 16, 
                 img_size: int = 224) -> None:
        """
        Args:
            embed_dim (int): Dimension of the encoder output embeddings.
            decoder_embed_dim (int): Dimension used in the decoder.
            depth (int): Number of transformer blocks in the decoder.
            num_heads (int): Number of attention heads in the decoder.
            patch_size (int): Size of each image patch (assumed square).
            img_size (int): Height/width of the input image (assumed square).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.img_size = img_size

        # Project encoder output to decoder embedding dimension
        self.proj = nn.Linear(embed_dim, decoder_embed_dim)

        # Learnable mask token 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Initialize a Vision Transformer for decoding (without a classification head)
        self.decoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=decoder_embed_dim,
            depth=depth, num_heads=num_heads, num_classes=0, mlp_ratio=4.0, qkv_bias=True
        )
        # Head to reconstruct flattened patch pixels (patch_size * patch_size * 3)
        self.reconstruction_head = nn.Linear(decoder_embed_dim, patch_size * patch_size * 3)

    def forward(self, 
                encoded_features: Tensor, 
                visible_indices: Tensor, 
                masked_indices: Tensor) -> Tensor:
        """
        Forward pass of the MAE decoder.

        Args:
            encoded_features (Tensor): Encoder output of shape (B, num_visible, embed_dim).
            visible_indices (Tensor): Indices of visible patches, shape (B, num_visible).
            masked_indices (Tensor): Indices of masked patches, shape (B, num_masked).

        Returns:
            Tensor: Reconstructed patches of shape (B, num_total_patches, patch_size*patch_size*3).
        """
        batch_size, num_visible_patches, _ = encoded_features.shape
        num_total_patches = (self.img_size // self.patch_size) ** 2

        # Create a full token tensor for all patches (initialized to zeros)
        full_tokens: Tensor = torch.zeros(batch_size, num_total_patches, self.decoder_embed_dim, 
                                            device=encoded_features.device)
        # Project visible tokens to the decoder embedding dimension
        encoded_features = self.proj(encoded_features)

        # Scatter the visible tokens into their corresponding positions
        full_tokens.scatter_(
            1,
            visible_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim),
            encoded_features
        )

        # Expand the mask token and scatter into masked positions
        mask_tokens: Tensor = self.mask_token.expand(batch_size, masked_indices.shape[1], -1)
        full_tokens.scatter_(
            1, 
            masked_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim), 
            mask_tokens
        )

        # Process the full token sequence with the decoder's transformer blocks and normalization
        decoder_features: Tensor = self.decoder.blocks(full_tokens)
        decoder_features = self.decoder.norm(decoder_features) 

        # Reconstruct the patches from decoder features
        reconstructed_patches: Tensor = self.reconstruction_head(decoder_features)
        return reconstructed_patches

class MaskedAutoEncoder(nn.Module):
    """Full Masked Autoencoder (MAE) that wraps the encoder and decoder."""
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 encoder_embed_dim=768, 
                 encoder_depth=12, 
                 encoder_num_heads=12,
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 decoder_num_heads=8
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2 

        self.encoder = MAEEncoder(
            img_size=img_size, patch_size=patch_size, embed_dim=encoder_embed_dim,
            depth=encoder_depth, num_heads=encoder_num_heads
        )

        self.decoder = MAEDecoder(
            embed_dim=encoder_embed_dim, decoder_embed_dim=decoder_embed_dim, depth=decoder_depth,
            num_heads=decoder_num_heads, 
        )

    def forward(self, x, mask_ratio=0.75):
        """
        Forward pass:
          - Extracts ground truth patches.
          - Encodes the image with random masking.
          - Decodes to reconstruct all patches.
          - Computes reconstruction loss (MSE) only on masked patches.
        """

        # Get target_patches from the original image
        target_patches = extract_patches(x, self.patch_size)

        # Encode image with masking
        encoded_features, visible_indices, masked_indices = self.encoder(x, mask_ratio)

        # Decode to reconstruct patches 
        reconstructed_patches = self.decoder(encoded_features, visible_indices, masked_indices)

        # Create a boolean mask of shape (B, num_patches) that is True for masked positions 
        batch_size, num_total_patches, _ = target_patches.shape
        mask = torch.zeros(batch_size, num_total_patches, dtype=torch.bool, device=x.device)
        mask.scatter_(1, masked_indices, True)

        return target_patches, reconstructed_patches, mask



# def test_mae_dummy():
#     import torch.nn.functional as F
#     # Define parameters
#     batch_size = 2
#     channels = 3
#     img_size = 224  # Should match the img_size used in your model
#     patch_size = 16

#     # Create a dummy batch of images
#     dummy_input = torch.randn(batch_size, channels, img_size, img_size)

#     # Instantiate the MAE model
#     mae_model = MaskedAutoEncoder(img_size=img_size, patch_size=patch_size)

#     # Forward pass through the MAE
#     target_patches, reconstructed_patches, mask = mae_model(dummy_input, mask_ratio=0.75)

#     # Print output shapes
#     print("Target patches shape:", target_patches.shape)
#     print("Reconstructed patches shape:", reconstructed_patches.shape)
#     print("Mask shape:", mask.shape)

#     # Optionally, compute a dummy reconstruction loss on the masked patches only
#     loss = F.mse_loss(reconstructed_patches[mask], target_patches[mask])
#     print("Dummy reconstruction loss:", loss.item())

# if __name__ == "__main__":
#     test_mae_dummy()

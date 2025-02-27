import torch 
import torch.nn as nn
import clip

# Convert an image into patches using nn.Conv2d
class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 patch_size:int=16,
                 in_channels:int=3,
                 embed_dim:int=768
                 ):
        
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2 # total number of patches

        # Use Conv2D to extract patches 
        # (B, C, W, H) -> (B, embed_dim, patch_size, patch_size)
        self.proj = nn.Conv2d(in_channels, 
                              embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)

    def forward(self, x:torch.Tensor):
        batch_size, channel, height, width = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2) # (batch_size, num_patches, embed_dim)
        return x 

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for dot-product attention

        # Separate linear layers for Q, K, and V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Final projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        # Compute Q, K, V separately
        q = self.W_q(x).reshape(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = self.W_k(x).reshape(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).reshape(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_weights = attn_scores.softmax(dim=-1)

        # Apply attention to values
        out = (attn_weights @ v)  # (B, num_heads, N, head_dim)

        # Merge heads back
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_patches, embed_dim)  # (B, N, embed_dim)

        return self.out_proj(out)  # Final linear projection
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        # Multi-Head Self attention
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)

        # MLP 
        self.ln_2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x):
        # Pre-norma + MHA + residual
        x = x + self.mha(self.ln_1(x))

        # Pre-norm + FFN + residual
        x = x + self.mlp(self.ln_2(x))

        return x 

class ImageEncoder(nn.Module):
    def __init__(self, 
                 img_size:int=224,
                 patch_size:int=16,
                 embed_dim:int=256,
                 num_heads:int=8,
                 depth:int=12,
                 mask_ratio:float=0.75
                 ):
        
        super().__init__()
        self.patch_embd = PatchEmbedding(img_size, patch_size, in_channels=3, embed_dim=embed_dim)
        self.num_patches = self.patch_embd.num_patches
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])

    
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape

        # convert image into patch embeddings 
        x = self.patch_embd(x) # (batch_size, num_patches, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Apply random masking (remove some patches)
        num_keep = int(self.num_patches * (1 - self.mask_ratio))
        mask = torch.rand(batch_size, self.num_patches).to(x.device)  # Generate random mask values
        mask_indices = mask.argsort(dim=1)[:, :num_keep]  # Get indices of patches to keep
        x_masked = torch.gather(x, 1, mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        for layer in self.encoder_layers:
            x_masked = layer(x_masked)

        return x_masked, mask_indices

"""
Fourier Positional Embedding is used in SAM to encoder (x,y) coorinates of 
points and boxes. 
"""

class FourierPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.freq_bands = torch.linspace(1.0, 64.0, steps=embed_dim // 4)

    def forward(self, coords):
        """
        Computes 2D Fourier positional embeddings. 
        """
        batch_size, num_coords, _ = coords.shape
        coords = coords.unsqueeze(-1) * self.freq_bands.to(coords.device)  # Shape: (B, N, 2, F)
        coords = torch.cat([coords.sin(), coords.cos()], dim=-1)
        return coords.view(batch_size, num_coords, -1)

class SparsePromptEncoder(nn.Module):
    """
    Encodes sparse prompts: Points, Boxes, and Text.
        - Points & Boxes: Encoded using positional embeddings.
        - Text: Encoded using the CLIP text encoder
    """
    def __init__(self, 
                 embed_dim=256, 
                 clip_model_name="ViT-B/32",
                 device="cpu"
                 ):
        super().__init__()
        self.embed_dim = embed_dim

        # Load CLIP model for text encoding
        self.device = device
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.text_proj = nn.Linear(512, embed_dim)

        # Fourier pos encoding
        self.fourier_pos_embed = FourierPositionalEmbedding(embed_dim)

        # Learnable embeddings for different prompt types
        self.point_embedding = nn.Embedding(2, embed_dim)  # Foreground (1) / Background (0)
        self.box_embedding = nn.Linear(4, embed_dim) # (x1, y1, x2, y2)

    def forward(self, points=None, boxes=None, text_prompts=None):
        """
        Forward pass:
            - points: Tensor of shape (batch_size, number_of_points, 3) where 3 = (x, y, type)
            - boxes: Tensor of shape (batch_size, 4) where 4 = (x1, y1, x2, y2)
            - text_prompts: List of text strings (["segment the dog", "find the car"])

            Returns:
            - Combined sparse prompt embeddings (batch_size, num_prompts, embed_dim)
        """

        embeddings = []
        if points is not None:
            # points.shape -> (batch_size, num_points_per_img, 3)
            # Each point has three values -> x_coor, y_coor, label (foreground/background)
            
            coordinates = points[:,:,:2] # :(first) -> select all batches, :(second) -> select all points, :2(third) -> select only the first two values (x,y)
            pos_embeds = self.fourier_pos_embed(coordinates) # Fourier encodings for (x,y)
            type_embeds = self.point_embedding(points[:, :, 2].long()) # Foreground/Background
            # pointe_embds -> (batch_size, num_points, embed_dim)
            point_embds = pos_embeds + type_embeds
            embeddings.append(point_embds)

        if boxes is not None:
            # box_embeds -> (batch_size, embed_dim)
            box_embeds = self.box_embedding(boxes) #(batch_size, 4) -> (batch_size, embed_dim)
            # (batch_size, embed_dim) -> (batch_size, 1, embed_dim)
            box_embeds = box_embeds.unsqueeze(1)
            embeddings.append(box_embeds) 
         
        if text_prompts is not None:
            text_inputs = clip.tokenize(text_prompts).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)  # (B, 512)
            # text_embeds -> (batch_size, embed_dim) 
            text_embeds = self.text_proj(text_features)  # Project CLIP output to embed_dim
            # (batch_size, embed_dim) -> (batch_size, 1, embed_dim)
            text_embeds = text_embeds.unsqueeze(1)
            embeddings.append(text_embeds)

        """
        embeddings = [
                        point_embeds,  # Shape: (batch_size, number_of_points, embed_dim)
                        box_embeds.unsqueeze(1),  # Shape: (batch_size, 1, embed_dim)
                        text_embeds.unsqueeze(1)  # Shape: (batch_size, 1, embed_dim)
                    ]
        """

        if len(embeddings) > 0:
            # (batch_size, number_of_points + 1 + 1, embed_dim) = #(batch_size, num_prompts, embed_dim)
            return torch.cat(embeddings, dim=1) 
        else:
            return None
        
class DensePromptEncoder(nn.Module):
    """
    Encodes dense prompts (masks) using convolutions
    - Downsamples mask using convolutions
    - Addes learned embeddings to refine segmentation
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # Downsampling convs for masks
        self.conv_1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, embed_dim, kernel_size=1) # Final 1x1 conv to match embed_dim

        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, mask):
        embed = self.gelu(self.conv1(mask))
        embed = self.gelu(self.conv2(embed))
        embed = self.ln(self.final_conv(embed))
        return embed
    

"""
Mask decoder takes in two types of embeddings 
    1. Image Embeddings + Dense Prompt Embeddings
    2. Prompt Embeddings 
"""
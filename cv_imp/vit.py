import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class MLP(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim,
                 dropout_proba=0.
                 ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_proba)
        self.linear_2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x 

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads=8,
                 dropout_proba=0.
                 ):
        super().__init__()
        assert dim // num_heads, "dim must be divisible by num_heads"
        dim_head = dim // num_heads
        inner_dim = num_heads * dim_head# dimension after splitting into multiple attention heads

        # if project_out is True, a linear layer will be applied to combine output of all heads 
        # back to the original input dimension
        # if flase, the output is left unchanged
        project_out = not(num_heads == 1 and dim_head == dim) 
        
        self.num_heads = num_heads
        self.scaling_factor = 1 / (math.sqrt(dim_head))

        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_proba)
        self.attend = nn.Softmax(dim=-1)
        
        """
        Let's say:
            dim = 512
            dim_head = 64
            num_heads = 8

        nn.Linear() maps input 
            512 -> 64 * 8 
        
        input: [batch_size, seq_len, dim]
        output: [batch_size, seq_len, dim_head * num_heads * 3]
        
        """

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if project_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout_proba)
            ) 
        else:
            self.to_out = nn.Identity()

    
    def forward(self, x):
        # [batch_size, seq_len, dim] => [batch_size, seq_len, dim]
        x = self.layer_norm(x)
        # [batch_size, seq_len, dim] => [batch_size, seq_len, dim_head * num_heads * 3]
        x = self.to_qkv(x)
        q,k,v = x.chunk(3, dim=-1)
        
        # [batch_size, seq_len, dim_head * num_heads] => [batch_size, num_heads, seq_len, dim_head]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # [batch_size, num_heads, seq_len, dim_head] => [batch_size, num_heads, seq_len, seq_len]
        scaled_dot_products = torch.matmul(q, k.transpose(-1,-2)) * self.scaling_factor
        attn_scores = self.dropout(self.attend(scaled_dot_products))

        out = torch.matmul(attn_scores, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

# num_heads = 8
# batch_size, seq_len, dim = 64, 10, 512 
# random_tensor = torch.randn(batch_size, seq_len, dim)
# layer = MultiHeadAttention(dim, num_heads)
# layer(random_tensor)

class TransformerLayer(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 mlp_hidden_dim,
                 dropout_proba=0.
                 ):
        super().__init__()
        self.mha = MultiHeadAttention(dim, num_heads=num_heads, dropout_proba=dropout_proba)
        self.mlp = MLP(dim, mlp_hidden_dim, dropout_proba=dropout_proba)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.mha(self.layer_norm(x)) + x 
        x = self.mlp(self.layer_norm(x)) + x 
        return x 

class Transformer(nn.Module):
    def __init__(self, 
                 dim,
                 num_transformer_layers,
                 num_heads,
                 mlp_hidden_dim,
                 dropout_proba=0.
                 ):
        super().__init__()
        self.num_transformer_layers = num_transformer_layers
        self.layer_fn = TransformerLayer(dim, num_heads, mlp_hidden_dim, dropout_proba)

    def forward(self, x):
        for _ in range(self.num_transformer_layers):
            x = self.layer_fn(x)
        return x 

class PatchEmbedding(nn.Module):
    def __init__(self,
                 patch_height,
                 patch_width, 
                 num_channels,
                 dim
                 ):
        super().__init__()
        patch_dim = num_channels * patch_height * patch_width
        self.rearrange = lambda x: rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        self.layer_norm_1 = nn.LayerNorm(patch_dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.linear = nn.Linear(patch_dim, dim)

    def forward(self, image):
        x = self.rearrange(image)   # Rearrange the input into patches
        x = self.layer_norm_1(x)  # Apply layer normalization to each patch
        x = self.linear(x)      # Apply the linear projection
        x = self.layer_norm_2(x)  # Apply layer normalization to the output
        return x 


class ViT(nn.Module):
    def __init__(self, 
                 *,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 num_transformer_layers,
                 num_heads,
                 mlp_hidden_dim,
                 pool="cls",
                 num_channels=3,
                 dropout_proba=0.,
                 emb_dropout_proba=0.
                 ):
        
        super().__init__()
        
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        assert image_height % patch_height == 0, 'image height must be divisible by patch height'
        assert image_width % patch_width == 0, 'image width must be divisible by patch width'
        assert pool in {'cls', 'mean'}, 'pool type must be ither cls (cls token) or mean (mean pooling)'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_embedding_layer = PatchEmbedding(patch_height, patch_width, num_channels,dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout_proba)

        self.transformer = Transformer(dim=dim, 
                                       num_transformer_layers=num_transformer_layers, 
                                       num_heads=num_heads, 
                                       mlp_hidden_dim=mlp_hidden_dim,
                                       dropout_proba=dropout_proba
                                       )
        
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, image):
        x = self.patch_embedding_layer(image)
        batch_size,num_patches, embedding_dim = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(num_patches + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == "mean" else x[:,0]
        x = self.to_latent(x)
        return self.mlp_head(x)



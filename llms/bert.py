import torch.nn as nn
import torch 
import math
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEmbedding, self).__init__()

        # Create a tensor of shape (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Calculate positional embeddings using broadcasting
        pos_embed = torch.zeros(max_len, d_model)
        pos_embed[:, 0::2] = torch.sin(pos * div_term)
        pos_embed[:, 1::2] = torch.cos(pos * div_term)

        # Register as a buffer to avoid updating gradients
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))

    def forward(self) -> torch.Tensor:
        return self.pos_embed

class BertEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 seq_len: int,
                 dropout: float=0.5
                 ):
        super(BertEmbedding, self).__init__()
        self.embed_size = embed_size

        self.token_embed = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # segment 0: first sentence in the input pair
        # segment 1: second sentence in the input pair
        self.segment_embed = torch.nn.Embedding(2, embed_size, padding_idx=0)  
        self.pos_embed = PositionalEmbedding(d_model=embed_size, max_len=seq_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, segment_label) -> torch.Tensor:
        x = self.token_embed(sequence) + self.pos_embed() + self.segment_embed(segment_label)
        return self.dropout(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 num_heads: int,
                 d_model: int, 
                 dropout: float=0.1
                 ):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model needs to be divisible by num_heads"

        self.d_k = d_model // num_heads # dim per head
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Split the heads: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2,-1)) / self.scale # [batch_size, num_heads, seq_len, seq_len]

        # apply mask (if any)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        # Concatenate heads: [batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # Final linear projection to get the output: [batch_size, seq_len, d_model]
        output = self.out_proj(attention_output)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, 
                 d_model:int, 
                 hidden_dim:int, 
                 dropout:float
                 ):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        out = self.gelu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 ffn_hidden_dim:int,
                 dropout:float
                 ):
        super(EncoderBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(num_heads=num_heads,d_model=d_model,dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model=d_model, hidden_dim=ffn_hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask) -> torch.Tensor:
        # x: (batch_size, max_len, d_model)
        # mask: (batch_size, 1, 1, max_len)
        # output: (batch_size, max_len, d_model)
        
        # MHA sublayer with residual conneciton
        attn_out, attn_weights = self.mha(x, x, x, mask)
        attn_out = self.dropout(attn_out)
        out_1 = self.layer_norm(x + attn_out)

        # Feed Forward Sublayer with residual connection
        ffn_out = self.ffn(out_1)
        ffn_out = self.dropout(ffn_out)
        out_2 = self.layer_norm(out_1 + ffn_out)

        return out_2


class BERT(nn.Module):
    def __init__(self, 
                 vocab_size:int,
                 d_model:int=768,
                 num_encoder_blks: int=12,
                 num_heads:int=12,
                 seq_len: int=64,
                 dropout:float=0.1
                 ):
        super(BERT,self).__init__()
        self.d_model = d_model
        self.num_encoder_blks = num_encoder_blks
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.ffn_hidden = d_model * 4 

        self.embedding = BertEmbedding(vocab_size=vocab_size, embed_size=d_model, seq_len=seq_len)
        self.encoder_blks = nn.ModuleList([EncoderBlock(d_model=d_model,num_heads=num_heads,ffn_hidden_dim=self.ffn_hidden,dropout=dropout) for _ in range(num_encoder_blks)])


    def forward(self, x, segment_label) -> torch.Tensor:
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embed
        x = self.embedding(x, segment_label)

        # pass x through encoder blocks
        for encoder in self.encoder_blks:
            x = encoder(x, mask)
        
        return x # Output shape: (batch_size, seq_len, d_model)
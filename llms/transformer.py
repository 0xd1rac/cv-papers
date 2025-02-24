import torch
import torch.nn as nn
import math
from typing import Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 n_embd:int, 
                 n_head:int, 
                 dropout:float=0.1
                 ):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.W_Q = nn.Linear(n_embd, n_embd)
        self.W_K = nn.Linear(n_embd, n_embd)
        self.W_V = nn.Linear(n_embd, n_embd)

        self.out = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.scaling_factor = math.sqrt(self.head_dim)
    
    def forward(self, 
                x:torch.Tensor, 
                context:torch.Tensor=None, 
                mask:torch.Tensor=None
                )-> Tuple[torch.Tensor, torch.Tensor]:

        if context is None:
            context = x 

        batch_size, seq_len, _ = x.size()
         # Linear projections for Q, K, V
        q = self.W_Q(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_K(context).view(batch_size, context.size(1), self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_V(context).view(batch_size, context.size(1), self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q,k.transpose(-1,-2)) / self.scaling_factor

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.contiguous().view(batch_size, seq_len, self.n_embd)
        output = self.out(attn_out)
        return output, attn_weights
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, 
                 n_embd:int, 
                 d_ff:int, 
                 dropout:float=0.1
                 ):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(n_embd, d_ff)
        self.fc2 = nn.Linear(d_ff, n_embd)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 


class EncoderBlock(nn.Module):
    def __init__(self, 
                 n_embd:int, 
                 n_head:int, 
                 d_ff:int, 
                 dropout:float=0.1
                 ):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(n_embd, n_head, dropout)
        self.ffn = FeedForwardNetwork(n_embd, d_ff, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor=None
                ) -> torch.Tensor:
        # MHA
        attn_output, _ = self.mha(x, mask)
        
        # add and norm
        x = self.ln1(x + attn_output)

        # ffn
        ffn_output = self.ffn(x)
        
        # add and norm
        x = self.ln2(x + ffn_output)

        return x 

class DecoderBlock(nn.Module):
    def __init__(self, 
                 n_embd:int, 
                 n_head:int, 
                 d_ff:int, 
                 dropout:float=0.1
                 ):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(n_embd, n_head,dropout)
        self.cross_attn =  MultiHeadAttention(n_embd, n_head,dropout)
        self.ffn = FeedForwardNetwork(n_embd,d_ff, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                enc_output: torch.Tensor, 
                src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None
               ) -> torch.Tensor:

        # Self attention (causal masking)
        self_attn_output, _ = self.self_attn(x, mask=tgt_mask)
        x = self.ln1(x + self_attn_output)  # add and norm

        # cross-attention
        cross_attn_output, _ = self.cross_attn(x, context=enc_output, mask=src_mask)
        x = self.ln2(x + cross_attn_output)# add and norm

        # Feed-forward network
        ffn_output = self.ffn(x) 
        x = self.ln3(x + ffn_output) # add and norm
        return x 

class Encoder(nn.Module):
    def __init__(self, 
                 vocab_size:int, 
                 n_embd:int, 
                 n_layer:int,
                 n_head:int,
                 d_ff:int, 
                 max_seq_len:int,
                 dropout:float=0.1
                 ):
        super(Encoder,self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(max_seq_len, n_embd)
        self.layers = nn.ModuleList([
            EncoderBlock(n_embd, n_head, d_ff, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, 
                src:torch.Tensor, 
                src_mask:torch.Tensor=None
                )-> torch.Tensor:
        seq_len = src.size(1)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=src.device).unsqueeze(0)

        # token + pos embeddings
        x = self.tok_emb(src) + self.pos_embd(pos_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.ln_f(x)
    
class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size:int, 
                 n_embd:int, 
                 n_layer:int,
                 n_head:int,
                 d_ff:int, 
                 max_seq_len:int,
                 dropout:float=0.1
                 ):
        super(Decoder, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(max_seq_len, n_embd)
        self.layers = nn.ModuleList(
            [DecoderBlock(n_embd, n_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                tgt:torch.Tensor, 
                enc_output:torch.Tensor,
                src_mask:torch.Tensor=None,
                tgt_mask:torch.Tensor=None
                )->torch.Tensor:
        seq_len = tgt.size(1)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=tgt.device).unsqueeze(0)

        # token + pos embeddings
        x = self.tok_emb(tgt) + self.pos_embd(pos_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return self.ln_f(x)

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size:int,
                 tgt_vocab_size:int,
                 n_embd:int,
                 n_layer:int,
                 n_head:int,
                 d_ff:int,
                 max_seq_len:int,
                 dropout:float=0.1
                 ):
        super(Transformer, self).__init__() 
        self.encoder = Encoder(src_vocab_size, n_embd, n_layer, n_head, d_ff, max_seq_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, n_embd, n_layer, n_head, d_ff, max_seq_len, dropout)
        self.out_proj = nn.Linear(n_embd, tgt_vocab_size)
    
    def forward(self, 
                src:torch.Tensor,
                tgt:torch.Tensor,
                src_mask:torch.Tensor=None,
                tgt_mask:torch.Tensor=None
                ) -> torch.Tensor:
        enc_output = self.encoder(src, src_mask)
        dec_ouput = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # final projection to the vocab size(logits)
        logits = self.out_proj(dec_ouput)
        return logits
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 n_embd:int,
                 n_head:int,
                 attn_pdrop:float
                 ):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.head_dim = n_embd // n_head

        self.W_Q, self.W_K, self.W_V = nn.Linear(n_embd, n_embd), nn.Linear(n_embd, n_embd), nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(attn_pdrop)
        self.out = nn.Linear(n_embd, n_embd)
        self.scaling_factor = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.size()
        # [batch_size, seq_len, n_embd] -> [batch_size, seq_len, n_head, head_dim] -> [batch_size, n_head, seq_len, head_dim]
        q  = self.W_Q(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        k  = self.W_K(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        v  = self.W_V(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)

        # attn_scores: [batch_size, n_head, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-1,-2)) / self.scaling_factor

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        
        # concat all the head back together
        attn_out = attn_out.contiguous().view(batch_size, seq_len, n_embd)
        output = self.out(attn_out)
        return output, attn_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, 
                 n_embd:int, 
                 n_inner:int, 
                 resid_pdrop:float
                ):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(n_embd, n_inner)
        self.fc2 = nn.Linear(n_inner, n_embd)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(resid_pdrop)
    
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 

class DecoderBlock(nn.Module):
    def __init__(self, 
                 n_embd:int, 
                 n_head:int,
                 n_inner:int,
                 attn_pdrop:float,
                 resid_pdrop:float,
                 layer_norm_epsilon:float
                 ):
        super(DecoderBlock,self).__init__()

        self.ln1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.attn = MultiHeadAttention(n_embd, n_head, attn_pdrop)
        self.ln2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.ffn = FeedForwardNetwork(n_embd, n_inner, resid_pdrop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        attn_ouput, _ = self.attn(x, mask)
        x = self.ln1(attn_ouput + x )

        ffn_output = self.ffn(x)
        x = self.ln2(ffn_output + x)
        return x
    

class GPTv1(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 n_positions:int, # seq_len
                 n_embd:int=768,
                 n_layer:int=12,
                 n_head:int=12,
                 n_inner:int=3072,
                 attn_pdrop:float=0.1,
                 embd_pdrop:float=0.1,
                 resid_pdrop:float=0.1,
                 layer_norm_epsilon:float=1e-5
                 ):
        super(GPTv1, self).__init__()

        if n_inner is None:
            n_inner = 4 * n_embd

        self.wte = nn.Embedding(vocab_size, n_embd) # word embeddings
        self.wpe = nn.Embedding(n_positions, n_embd) # position embeddings
        self.embd_dropout = nn.Dropout(embd_pdrop)

        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head, n_inner, attn_pdrop, resid_pdrop, layer_norm_epsilon) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd, eps=layer_norm_epsilon) # final layer normalization
        self.clf = nn.Linear(n_embd, vocab_size, bias=False) # output layer (logits)

    def forward(self, input_ids: torch.Tensor,mask: torch.Tensor=None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Embedding + Positional Embedding + Dropout
        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.embd_dropout(x)

        # Pass through each transformer decoder block
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.clf(x)
        return logits

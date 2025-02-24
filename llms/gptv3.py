import torch.nn as nn
import torch 
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout:float):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
    
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scaling_factor = 1 / math.sqrt(self.d_k)
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        q = self.W_q(x)  # (batch_size, seq_len, d_model)
        k = self.W_k(x)  # (batch_size, seq_len, d_model)
        v = self.W_v(x)  # (batch_size, seq_len, d_model)

        # Reshape and transpose for attention
        # (batch_size, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        # (batch_size, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling_factor
        
        if mask is not None:
            # Expand mask for multiple heads
            # mask shape should be (batch_size, 1, seq_len, seq_len)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch_size, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, v)
        
        # (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.W_o(attn_output)
        return output

class LocallyBandedSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.scaling_factor = 1 / math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Initialize output tensor
        output = torch.zeros_like(q)
        
        # Compute attention for each position using only its window
        for i in range(seq_len):
            window_start = max(0, i - self.window_size)
            window_end = min(seq_len, i + self.window_size + 1)
            
            # Extract queries, keys, and values for the current window
            q_i = q[:, i:i+1, :, :]  # (batch_size, 1, n_heads, d_k)
            k_w = k[:, window_start:window_end, :, :]  # (batch_size, window_size*2+1, n_heads, d_k)
            v_w = v[:, window_start:window_end, :, :]  # (batch_size, window_size*2+1, n_heads, d_k)
            
            # Compute attention scores for the window
            attn_scores = torch.matmul(q_i, k_w.transpose(-2, -1)) * self.scaling_factor
            
            # Apply mask if provided
            if mask is not None:
                window_mask = mask[:, i:i+1, window_start:window_end]
                attn_scores = attn_scores.masked_fill(window_mask == 0, float('-inf'))
            
            # Compute attention weights and apply dropout
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Compute attention output for current position
            output[:, i:i+1, :, :] = torch.matmul(attn_weights, v_w)
        
        # Reshape and apply final linear transformation
        output = output.reshape(batch_size, seq_len, d_model)
        return self.W_o(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 

class DecoderBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_ff:int, window_size:int, dropout:float, use_sparse_attn:bool=False):
        super(DecoderBlock, self).__init__()

        if use_sparse_attn:
            self.attn = LocallyBandedSparseAttention(d_model, n_heads, window_size, dropout)
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None)->torch.Tensor:
        # Pre-normalization
        x_norm = self.ln1(x)
        attn_output = self.attn(x_norm, mask)
        x = x + attn_output

        # Post-normalization
        x_norm = self.ln2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        return x 

class GPTv3(nn.Module):
    def __init__(self, 
                 vocab_size:int, 
                 n_positions:int,
                 d_model:int, 
                 n_layers:int, 
                 n_heads:int, 
                 d_ff:int, 
                 window_size:int, 
                 attn_pdrop:float,
                 embd_pdrop:float, 
                 resid_pdrop:float,
                 layer_norm_epsilon:float,
                 ):
        super(GPTv3, self).__init__()

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(n_positions, d_model)
        self.embd_dropout = nn.Dropout(embd_pdrop)

        self.blocks = nn.ModuleList(
            [DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                window_size=window_size,
                dropout=attn_pdrop,
                use_sparse_attn=False
            ) for _ in range(n_layers)])
        
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.clf = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids:torch.Tensor, mask:torch.Tensor=None)->torch.Tensor:
        _, seq_len = input_ids.size()
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.embd_dropout(x)

        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.clf(x)
        return logits

if __name__ == "__main__":
    # Define constants
    VOCAB_SIZE = 1000
    SEQ_LEN = 64
    BATCH_SIZE = 32 
    D_MODEL = 768
    
    # Create dummy input data: random token IDs within the range [0, VOCAB_SIZE-1]
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    model = GPTv3(vocab_size=VOCAB_SIZE, 
                   n_positions=SEQ_LEN, 
                   d_model=D_MODEL, 
                   n_layers=6, 
                   n_heads=12, 
                   d_ff=2048, 
                   window_size=16, 
                   attn_pdrop=0.1, 
                   embd_pdrop=0.1, 
                   resid_pdrop=0.1, 
                   layer_norm_epsilon=1e-5)
    logits = model(input_ids)
    print(logits.shape)
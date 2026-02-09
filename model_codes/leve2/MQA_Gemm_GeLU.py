import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Multi-Query Attention (MQA) implementation.
    Key and value are shared across heads, only query is multi-head.
    """

    def __init__(self, n_embd, n_head, out_features, batch_size, seq_len, max_seqlen,dtype=torch.float16):
        super().__init__()
        assert n_embd % n_head == 0
        # Query projection: multi-head
        self.q_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        # Key/Value projection: single-head (shared)
        self.k_proj = nn.Linear(n_embd, n_embd // n_head, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_embd // n_head, dtype=dtype)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen, dtype=dtype))
        .view(1, 1, max_seqlen, max_seqlen))
        self.gemm = nn.Linear(n_embd, out_features, bias=False).to(dtype)
        self.n_head = n_head
        self.n_embd = n_embd
        self.batch_size=batch_size
        self.seq_len=seq_len

    def forward(self, x):

        # Multi-head query
        q = self.q_proj(x)
        q = q.view(self.batch_size, self.seq_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Shared key/value
        k = self.k_proj(x)  # (B, T, head_dim)
        v = self.v_proj(x)  # (B, T, head_dim)
        # Expand k/v for broadcasting to all heads
        k = k.unsqueeze(1).expand(self.batch_size, self.n_head, self.seq_len, self.n_embd // self.n_head)  # (B, n_head, T, head_dim)
        v = v.unsqueeze(1).expand(self.batch_size, self.n_head, self.seq_len, self.n_embd // self.n_head)  # (B, n_head, T, head_dim)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        bias = self.bias[:, :, :self.seq_len, :self.seq_len]
        att = att.masked_fill(bias == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.n_embd) # re-assemble all head outputs side by side
        
        # output projection
        y = self.c_proj(y)
        y = self.gemm(y)
        y = torch.nn.functional.gelu(y)  # Apply GeLU activation
        return y

def get_default_input_shapes():
    """
    Returns the default input shape for the model.
    The shape is (batch_size, seq_len, n_embd).
    """
    batch_size = 64
    seq_len = 256
    n_embd = 128
    return [batch_size, seq_len, n_embd]

def get_default_model_params_shapes():
    batch_size = 64
    seq_len = 256
    max_seqlen = 1024
    n_head = 8
    n_embd = 128
    out_features = 64
    return [n_embd, n_head, out_features, batch_size,seq_len, max_seqlen]

def get_inputs(batch_size, seq_len, n_embd, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*seq_len*n_embd) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.empty(batch_size, seq_len, n_embd, dtype=dtype)]
    else:
        return None


def get_model(n_embd, n_head,out_features, batch_size,seq_len, max_seqlen, dtype=torch.float16):
    return Model(n_embd, n_head, out_features, batch_size,seq_len,max_seqlen, dtype)

def get_real_inputs(batch_size=128, seq_len=512, n_embd=768, dtype=torch.float16):
    return [torch.randn(batch_size, seq_len, n_embd, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    n_embd = 128
    n_head = 8
    out_features = 64
    max_seqlen = 1024
    batch_size = 64
    seq_len = 256
    range_n_embd = [64, 512]
    range_n_head = [2, 4, 8, 16, 32, 64, 128]
    range_out_features = [8, 1024]
    range_max_seqlen = [512, 1024]
    range_batch_size = [1, 1024]
    range_seq_len = [1, 1024]
    dtypes = [int, int, int,int, int, int]
    
    return [n_embd, n_head, out_features, max_seqlen, batch_size, seq_len], \
           [range_n_embd, range_n_head, range_out_features, range_max_seqlen, range_batch_size, range_seq_len], dtypes

def split_shapes_into_input_and_model_params_shapes(n_embd, n_head, out_features, max_seqlen, batch_size, seq_len):
    return [batch_size, seq_len, n_embd], \
           [n_embd, n_head, out_features, batch_size, seq_len, max_seqlen]  # attn_pdrop and resid_pdrop are set to 0.0 by default

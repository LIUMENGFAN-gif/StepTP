import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Grouped-Query Attention (GQA) implementation.
    Query heads are divided into groups, each group shares its own key and value.
    """

    def __init__(self, n_embd, n_head, n_group, batch_size, seq_len, max_seqlen, dtype=torch.float16):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_group == 0, "n_head must be divisible by n_group"
        self.n_head = n_head
        self.n_group = n_group
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.group_size = n_head // n_group

        # Multi-head query
        self.q_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        # Grouped key/value
        self.k_proj = nn.Linear(n_embd, n_group * self.head_dim, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_group * self.head_dim, dtype=dtype)
        self.c_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen, dtype=dtype)).view(1, 1, max_seqlen, max_seqlen)
        )
        self.batch_size = batch_size
        self.seq_len = seq_len

    def forward(self, x):
        # Multi-head query
        q = self.q_proj(x).view(self.batch_size, self.seq_len, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        # Grouped key/value
        k = self.k_proj(x).view(self.batch_size, self.seq_len, self.n_group, self.head_dim)  # (B, T, n_group, head_dim)
        v = self.v_proj(x).view(self.batch_size, self.seq_len, self.n_group, self.head_dim)  # (B, T, n_group, head_dim)

        k = k.unsqueeze(2).expand(self.batch_size, self.seq_len, self.group_size, self.n_group, self.head_dim).contiguous().view(self.batch_size, self.seq_len, self.n_head, self.head_dim)
        v = v.unsqueeze(2).expand(self.batch_size, self.seq_len, self.group_size, self.n_group, self.head_dim).contiguous().view(self.batch_size, self.seq_len, self.n_head, self.head_dim)
        k = k.transpose(1, 2)  # 等价于 k.permute(0, 2, 1, 3)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, n_head, T, T)
        bias = self.bias[:, :, :self.seq_len, :self.seq_len]
        att = att.masked_fill(bias == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.n_embd)
        y = self.c_proj(y)
        y = torch.mean(y, dim=1, keepdim=True)
        return y

def get_default_input_shapes():
    batch_size = 64
    seq_len = 256
    n_embd = 128
    return [batch_size, seq_len, n_embd]

def get_default_model_params_shapes():
    batch_size = 64
    seq_len = 256
    max_seqlen = 1024
    n_head = 8
    n_group = 2  # 例如分2组
    n_embd = 128
    return [n_embd, n_head, n_group, batch_size, seq_len, max_seqlen]

def get_inputs(batch_size, seq_len, n_embd, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*seq_len*n_embd) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.empty(batch_size, seq_len, n_embd, dtype=dtype)]
    else:
        return None

def get_model(n_embd, n_head, n_group, batch_size, seq_len, max_seqlen, dtype=torch.float16):
    return Model(n_embd, n_head, n_group, batch_size, seq_len, max_seqlen, dtype)

def get_real_inputs(batch_size=128, seq_len=512, n_embd=768, dtype=torch.float16):
    return [torch.randn(batch_size, seq_len, n_embd, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    n_embd = 128
    n_head = 8
    n_group = 2
    max_seqlen = 1024
    batch_size = 64
    seq_len = 256
    range_n_embd = [64, 512]
    range_n_head = [2, 4, 8, 16, 32, 64, 128]
    range_n_group = [1, 16]
    range_max_seqlen = [512, 1024]
    range_batch_size = [1, 1024]
    range_seq_len = [1, 1024]
    dtypes = [int, int, int, int, int, int]
    return [n_embd, n_head, n_group, max_seqlen, batch_size, seq_len], \
           [range_n_embd, range_n_head, range_n_group, range_max_seqlen, range_batch_size, range_seq_len], dtypes

def split_shapes_into_input_and_model_params_shapes(n_embd, n_head, n_group, max_seqlen, batch_size, seq_len):
    return [batch_size, seq_len, n_embd], [n_embd, n_head, n_group, batch_size, seq_len, max_seqlen]

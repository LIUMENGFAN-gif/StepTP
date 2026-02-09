import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64, dtype=torch.float16):
        """
        Mamba Structured State Space model implementation for benchmarking.
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        :param dtype: Data type for parameters
        """
        super(Model, self).__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        self.dtype = dtype
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, dtype=dtype))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state, dtype=dtype))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state, dtype=dtype))
        self.register_buffer("Amask", torch.tril(torch.ones(self.block_len, self.block_len, dtype=bool), diagonal=0))

    def forward(self, X):
        """
        Forward pass implementing the SSD operation.
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :return: Output tensor Y and final state
        """
        # Rearrange into blocks/chunks
        X_blocks = X.contiguous().view(self.batch_size, self.seq_length// self.block_len, self.block_len, self.n_heads, self.d_head)
        A_blocks = self.A.contiguous().view(self.batch_size, self.seq_length // self.block_len, self.block_len, self.n_heads)
        B_blocks = self.B.contiguous().view(self.batch_size, self.seq_length // self.block_len, self.block_len, self.n_heads, self.d_state)
        C_blocks = self.C.contiguous().view(self.batch_size, self.seq_length // self.block_len, self.block_len, self.n_heads, self.d_state)

        # Replace rearrange(A_blocks, "b c l h -> b h c l") with permute
        A_blocks = A_blocks.transpose(1, 3).transpose(2, 3)
        
        # 1. Compute diagonal block outputs
        A_cumsum = torch.cumsum(A_blocks, dim=-1) # shape: self.batch_size, self.n_heads, self.seq_length//self.block_len, self.block_len
        A_segsum = A_cumsum.unsqueeze(-1) - A_cumsum.unsqueeze(-2)
        A_segsum = A_segsum.masked_fill(~self.Amask, float('-inf'))
        L = torch.exp(A_segsum)
        LX = torch.einsum("bhcls,bcshp->bclhp", L, X_blocks)
        BLX = torch.einsum("bclhn,bclhp->bclnp", B_blocks, LX)
        Y_diag = torch.einsum("bclhn,bclnp->bclhp", C_blocks, BLX)
        return Y_diag

        # # 2. Compute intra-chunk states
        # last = A_cumsum[:,:,:,-1].unsqueeze(-1)
        # decay_states = torch.exp((last - A_cumsum))
        # # First einsum: combine decay_states and X_blocks
        # decayed_X = torch.einsum("bhcl,bclhp->bchp", decay_states, X_blocks)
        # # Second einsum: combine B_blocks and decayed_X
        # states = torch.einsum("bclhn,bchp->bchpn", B_blocks, decayed_X)
        

        # # 3. Compute inter-chunk recurrence
        # initial_states = torch.zeros_like(states[:, :1])
        # states = torch.cat([initial_states, states], dim=1)

        # A_cumsum_temp = A_cumsum[:, :, :, -1] # shape: self.batch_size, self.n_heads, self.seq_length//self.block_len
        # zero = torch.zeros_like(A_cumsum_temp[:,:,:1])  # shape: self.batch_size, self.n_heads, 1
        # padded = torch.cat([zero, A_cumsum_temp], dim=-1)  # shape: self.batch_size, self.n_heads, self.seq_length//self.block_len+1
        # x_cumsum = torch.cumsum(padded, dim=-1)
        # x_segsum = x_cumsum.unsqueeze(-1) - x_cumsum.unsqueeze(-2)
        # T=self.seq_length//self.block_len+1
        # mask = torch.tril(torch.ones(T, T, dtype=bool), diagonal=0)
        # x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        # decay_chunk = torch.exp(x_segsum)
        # new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        # states = new_states[:, :-1]

        # # 4. Compute state-to-output conversion
        # state_decay_out = torch.exp(A_cumsum)
        # decayed_states = torch.einsum('bchpn,bhcl->bchpln', states, state_decay_out)
        # Y_off = torch.einsum('bclhn,bchpln->bclhp', C_blocks, decayed_states)

        # # Combine diagonal and off-diagonal terms
        # Y = Y_diag + Y_off 
        # return Y


def get_default_input_shapes():
    batch_size = 16
    seq_length = 128
    n_heads = 8
    d_head = 64
    return [batch_size, seq_length, n_heads, d_head]


def get_default_model_params_shapes():
    batch_size = 16
    seq_length = 128
    n_heads = 8
    d_head = 64
    d_state = 16
    block_len = 64
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]


def get_inputs(batch_size, seq_length, n_heads, d_head, dtype=torch.float16):
    return [torch.randn(batch_size, seq_length, n_heads, d_head, dtype=dtype)]


def get_model(batch_size, seq_length, n_heads, d_head, d_state, block_len=64, dtype=torch.float16):
    return Model(batch_size, seq_length, n_heads, d_head, d_state, block_len, dtype=dtype)

def get_real_inputs(batch_size, seq_length, n_heads, d_head, dtype=torch.float16):
    return [torch.randn(batch_size, seq_length, n_heads, d_head, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 16
    seq_length = 128
    n_heads = 8
    d_head = 64
    d_state = 16
    block_len = 64
    range_batch_size = [1, 1024]
    range_seq_length = [1, 2048]
    range_n_heads = [1, 16]
    range_d_head = [16, 512]
    range_d_state = [8, 256]
    range_block_len = [16, 128]
    dtypes = [int, int, int, int, int, int]
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len], \
           [range_batch_size, range_seq_length, range_n_heads, range_d_head, range_d_state, range_block_len], dtypes

def split_shapes_into_input_and_model_params_shapes(batch_size, seq_length, n_heads, d_head, d_state, block_len):
    return [batch_size, seq_length, n_heads, d_head],\
    [batch_size, seq_length, n_heads, d_head, d_state, block_len]
    
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

def get_model(shape=None,dtype=torch.float16):
    return Model(dtype)

def get_default_input_shapes():
    M = 1024
    N = 1024
    K = 32
    return [M,K, N]

def get_default_model_params_shapes():
    # No model parameters for this model
    return []

def get_inputs(M=16384, N=16384, K=32, dtype=torch.float16):
    A = torch.empty(M, K, dtype=dtype, device='meta')
    B = torch.empty(K, N, dtype=dtype, device='meta')
    return [A, B]

def get_real_inputs(M=16384, N=16384, K=32, dtype=torch.float16):
    A = torch.randn(M, K, dtype=dtype)
    B = torch.randn(K, N, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    M = 1024
    N = 1024
    K = 32
    range_M = [8, 1024]
    range_K = [8, 128]
    range_N = [128, 4096]
    dtype = [int, int, int]

    return [M, K, N], \
           [range_M, range_K, range_N], \
           dtype

def split_shapes_into_input_and_model_params_shapes(M, K, N):
    return [M, K, N], []  # No model parameters in this case
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, A: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return A * s

def get_model(shape=None, dtype=torch.float16):
    return Model(dtype)

def get_default_input_shapes():
    M = 1024
    N = 512
    return [M, N] 

def get_default_model_params_shapes():
    # No model parameters for this model
    return []

def get_inputs(M=4096, N=4096, s=3.14, dtype=torch.float16):
    A = torch.empty(M, N, dtype=dtype, device='meta')
    s = torch.tensor(s, dtype=dtype, device='meta')
    return [A, s]

def get_real_inputs(M=4096, N=4096,s=3.14, dtype=torch.float16):
    A = torch.randn(M, N, dtype=dtype)
    s = torch.tensor(s, dtype=dtype)
    return [A, s]

def set_default_shapes_ranges_and_dtypes():
    M = 1024
    N = 512
    s = 3.14
    range_M = [8, 1024]
    range_N = [8, 4096]
    range_s = [0.1, 10.0]  # Example range for scalar
    dtype = [int, int, float]
    return [M, N, s], \
           [range_M, range_N, range_s], \
           dtype

def split_shapes_into_input_and_model_params_shapes(M, N, s):
    return [M, N, s], []  # No model parameters in this case
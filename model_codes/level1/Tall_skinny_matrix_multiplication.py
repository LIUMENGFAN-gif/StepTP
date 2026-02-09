import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    Parameters:
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, dtype=torch.float16):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, A, B):
        return torch.matmul(A, B)

def get_default_input_shapes():
    M = 1024
    N = 16
    return [M, N]

def get_default_model_params_shapes():
    # No learnable parameters for Tall_skinny_matrix_multiplication
    return []

def get_inputs(M=16384, N=16, dtype=torch.float16):
    A = torch.empty(M, N, dtype=dtype, device='meta')
    B = torch.empty(N, M, dtype=dtype, device='meta')
    return [A, B]

def get_model(shape=None,dtype=torch.float16):
    return Model(dtype=dtype)

def get_real_inputs(M=16384, N=16, dtype=torch.float16):
    A = torch.randn(M, N, dtype=dtype)
    B = torch.randn(N, M, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    M = 1024
    N = 16
    range_M = [1, 1024]
    range_N = [1, 64]
    dtype = [int, int]

    return [M, N], \
           [range_M, range_N], \
           dtype

def split_shapes_into_input_and_model_params_shapes(M, N):
    return [M, N], []
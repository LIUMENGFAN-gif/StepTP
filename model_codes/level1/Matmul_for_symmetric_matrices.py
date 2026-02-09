import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices.
        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.
        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        A = (A + A.T) / 2 
        B = (B + B.T) / 2
        return torch.matmul(A, B)

def get_default_input_shapes():
    """
    Returns the default input shapes for the model.
    The shapes are (N, N) for both A and B.
    """
    N = 512
    return [N]

def get_default_model_params_shapes():
    return []

def get_inputs(N, dtype=torch.float16):
    A = torch.empty(N, N, dtype=dtype, device='meta')
    B = torch.empty(N, N, dtype=dtype, device='meta')
    return [A, B]

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(N=4096, dtype=torch.float16):
    A = torch.randn(N, N, dtype=dtype)
    B = torch.randn(N, N, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    N = 512
    range_N = [1, 1024]
    dtype = [int, int]
    
    return [N], \
           [range_N], \
           dtype

def split_shapes_into_input_and_model_params_shapes(N):
    return [N], []

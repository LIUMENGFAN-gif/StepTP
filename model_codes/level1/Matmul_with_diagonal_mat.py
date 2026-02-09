import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
        A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
        B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
        torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return torch.diag(A) @ B


def get_default_input_shapes():
    M = 1024
    N = 512
    return [N, M]

def get_default_model_params_shapes():
    return []

def get_inputs(N,M, dtype=torch.float16):
    A = torch.empty(N, dtype=dtype, device='meta')  # Diagonal elements
    B = torch.empty(N, M,dtype=dtype, device='meta')  # Second matrix
    return [A, B]

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(N=4096, M=4096, dtype=torch.float16):
    A = torch.randn(N, dtype=dtype)
    B = torch.randn(N, M, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    M = 1024
    N = 512
    range_N = [1, 4096]
    range_M = [1, 1024]
    dtype = [int, int]
    
    return [N, M], \
           [range_N, range_M], \
           dtype

def split_shapes_into_input_and_model_params_shapes(N, M):
    return [N, M], []



import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.
        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).
        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        A = torch.tril(A)
        B = torch.tril(B)
        return torch.tril(torch.matmul(A, B))

def get_default_input_shapes():
    """
    Returns the default input shapes for the model.
    The shapes are (N, N) for both A and B.
    """
    N = 1024
    return [N]

def get_default_model_params_shapes():
    return []

def get_inputs(N, dtype=torch.float16):
    """
    Generates lower triangular matrices for testing.
    Returns:
        list: A list containing two lower triangular matrices of shape (N, N).
    """
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
    N = 1024
    range_N = [1, 1024]
    dtype = [int, int]
    
    return [N], \
           [range_N], \
           dtype

def split_shapes_into_input_and_model_params_shapes(N):
    return [N], []  # No model parameters in this case
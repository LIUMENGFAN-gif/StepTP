import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return torch.matmul(A, B)

def get_inputs(M, K, dtype=torch.float16):
    A = torch.randn(M, K, dtype=dtype)
    B = torch.randn(K, 1, dtype=dtype)
    return [A, B]

def get_default_input_shapes():
    M = 256
    K = 131072
    return [M, K]

def get_default_model_params_shapes():
    return []

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(M=256, K=131072, dtype=torch.float16):
    A = torch.randn(M, K, dtype=dtype)
    B = torch.randn(K, 1, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    M = 256
    K = 131072
    range_M = [1, 1024]
    range_K = [1, 4096]
    dtype = [int, int]

    return [M, K], \
           [range_M, range_K], \
           dtype

def split_shapes_into_input_and_model_params_shapes(M, K):
    return [M, K], []
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with B transposed.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B.T)

def get_default_input_shapes():
    """
    Returns the default input shapes for the model.
    The shapes are (M, K) for A and (N, K) for B.
    """
    K = 512
    M = 1024
    N = 768
    return [M, K, N]

def get_default_model_params_shapes():
    return []

def get_inputs(M, K, N, dtype=torch.float16):
    """
    Generates input tensors for testing.

    Args:
        M: Number of rows for input tensor A.
        K: Number of columns for input tensor A / rows for input tensor B.
        N: Number of columns for input tensor B.
        dtype: Data type of the tensors.

    Returns:
        list: [A, B] where A is (M, K) and B is (N, K), both on device='meta'.
    """
    A = torch.empty(M, K, dtype=dtype, device='meta')
    B = torch.empty(N, K, dtype=dtype, device='meta')
    return [A, B]

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(M=1024, K=4096, N=2048, dtype=torch.float16):
    A = torch.randn(M, K, dtype=dtype)
    B = torch.randn(N, K, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    K = 512
    M = 1024
    N = 768
    range_M = [1, 4096]
    range_K = [1, 1024]
    range_N = [1, 1024]
    dtype = [int, int, int]
    
    return [M, K, N], \
           [range_M, range_K, range_N], \
           dtype

def split_shapes_into_input_and_model_params_shapes(M, K, N):
    return [M, K, N], []
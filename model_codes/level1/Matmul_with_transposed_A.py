import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A transposed.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).
        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A.T, B)

def get_default_input_shapes():
    """
    Returns the default input shapes for the model.
    The shapes are (K, M) for A and (K, N) for B.
    """
    K = 512
    M = 1024
    N = 768
    return [K, M, N]

def get_default_model_params_shapes():
    return []

def get_inputs(K, M, N, dtype=torch.float16):
    """
    Generates input tensors for testing.
    Returns:
        list: [A, B] where A is (K, M) and B is (K, N), both on device='meta'.
    """
    A = torch.empty(K, M, dtype=dtype, device='meta')
    B = torch.empty(K, N, dtype=dtype, device='meta')
    return [A, B]

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(K=4096, M=1024, N=2048, dtype=torch.float16):
    A = torch.randn(K, M, dtype=dtype)
    B = torch.randn(K, N, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    K = 512
    M = 1024
    N = 768
    range_K = [1, 1024]
    range_M = [1, 1024]
    range_N = [1, 4096]
    dtype = [int, int, int]
    
    return [K, M, N], \
           [range_K, range_M, range_N], \
           dtype

def split_shapes_into_input_and_model_params_shapes(K, M, N):
    return [K, M, N], []

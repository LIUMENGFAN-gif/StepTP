import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)

def get_default_input_shapes():
    """
    Returns the default input shapes for the model.
    The shape is (N, N) for both A and B.
    """
    N = 1024
    return [N]

def get_default_model_params_shapes():
    return []

def get_inputs(N, dtype=torch.float16):
    """
    Generates input tensors for testing.

    Args:
        N (int): The size of the square matrices.
        dtype: The data type of the tensors.

    Returns:
        list: [A, B] where both are (N, N) on device='meta'.
    """
    A = torch.empty(N, N, dtype=dtype, device='meta')
    B = torch.empty(N, N, dtype=dtype, device='meta')
    return [A, B]

def get_model(shape=None, dtype=torch.float16):
    """
    Returns an instance of the model.
    """
    return Model()

def get_real_inputs(N=2048, dtype=torch.float16):
    A = torch.randn(N, N, dtype=dtype)
    B = torch.randn(N, N, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    N = 1024
    range_N = [1, 1024]
    dtype = [int]
    return [N], \
           [range_N], \
           dtype

def split_shapes_into_input_and_model_params_shapes(N):
    return [N], []  # All shapes are input shapes, no model parameters
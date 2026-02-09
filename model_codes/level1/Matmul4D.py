import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs 4D tensor-matrix multiplication:
    C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
    A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
    B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
    torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return torch.einsum("bijl,lk->bijk", A, B)


def get_default_input_shapes():
    """
    Returns the default input shapes for the model.
    The shapes are (batch_size, i, j, l) for A and (l, k) for B.
    """
    b = 16
    i = 256
    j = 512
    l = 256
    k = 128
    return [b, i, j, l, k]

def get_default_model_params_shapes():
    return []

def get_inputs(b, i, j, l, k, dtype=torch.float16):
    A = torch.empty(b, i, j, l, dtype=dtype, device='meta')
    B = torch.empty(l, k, dtype=dtype, device='meta')
    return [A, B]

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(b=16, i=256, j=512, l=256, k=768, dtype=torch.float16):
    A = torch.randn(b, i, j, l, dtype=dtype)
    B = torch.randn(l, k, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    b = 16
    i = 256
    j = 512
    l = 256
    k = 128
    range_b = [1, 1024]
    range_i = [1, 2048]
    range_j = [1, 2048]
    range_l = [1, 2048]
    range_k = [1, 2048]
    dtype = [int, int, int, int, int]

    return [b, i, j, l, k], \
           [range_b, range_i, range_j, range_l, range_k], \
           dtype

def split_shapes_into_input_and_model_params_shapes(b, i, j, l, k):
    return [b, i, j, l, k], []
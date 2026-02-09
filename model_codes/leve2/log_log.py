import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B) -> torch.Tensor:
        E=torch.log(A)
        F=torch.log(B)
        return torch.bmm(E,F)

def get_inputs(batch_size, m, k, n, dtype=torch.float16):
    A = torch.empty(batch_size, m, k, dtype=dtype, device='meta')
    B = torch.empty(batch_size, k, n, dtype=dtype, device='meta')
    # Return a list of tensors
    return [A, B]

def get_default_input_shapes():
    batch_size = 128
    m = 128
    k = 256
    n = 512
    return [batch_size, m, k, n]

def get_default_model_params_shapes():
    return []

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(batch_size=128, m=128, k=256, n=512, dtype=torch.float16):
    A = abs(torch.randn(batch_size, m, k, dtype=dtype))
    B = abs(torch.randn(batch_size, k, n, dtype=dtype))
    # Return a list of tensors
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 128
    m = 128
    k = 256
    n = 512
    range_batch_size = [1, 1024]
    range_m = [1, 4089]
    range_k = [1, 4089]
    range_n = [1, 4089]
    dtype = [int, int, int, int]
    return [batch_size, m, k, n], \
           [range_batch_size, range_m, range_k, range_n], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, m, k, n):
    return [batch_size, m, k, n], []  # No model parameters in this case
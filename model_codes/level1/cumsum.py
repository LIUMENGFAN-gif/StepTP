import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
        dtype (torch.dtype, optional): Data type for the input. Defaults to torch.float16.
    """
    def __init__(self, dim, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x):
        return torch.cumsum(x, dim=self.dim)

def get_model(dim=1, dtype=torch.float16):
    return Model(dim, dtype)

def get_default_input_shapes():
    batch_size = 128
    input_shape = 4000
    return [batch_size, input_shape]

def get_default_model_params_shapes():
    dim = 1
    return [dim]

def get_inputs(batch_size=128, input_shape=4000, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_shape)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, input_shape, dtype=dtype)
        return [x]
    else:
        return None

def get_real_inputs(batch_size=128, input_shape=4000, dtype=torch.float16):
    return [torch.randn(batch_size, input_shape, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 128
    input_shape = 4000
    dim = 1
    range_batch_size = [1, 4096]
    range_input_shape = [1, 4096]
    range_dim = [0, 1]  # Cumulative sum can be along any dimension
    dtype = [int, int, int]
    return [batch_size, input_shape, dim], \
           [range_batch_size, range_input_shape, range_dim], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, input_shape, dim):
    return [batch_size, input_shape], [dim]
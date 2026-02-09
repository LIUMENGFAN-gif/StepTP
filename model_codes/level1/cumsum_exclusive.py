import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """

    def __init__(self, dim, dtype=torch.float16):
        super(Model, self).__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x):
        exclusive_cumsum = torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), x), dim=self.dim)[:-1]
        return torch.cumsum(exclusive_cumsum, dim=self.dim)

def get_default_input_shapes():
    batch_size = 128
    input_shape = 1000
    return [batch_size, input_shape]

def get_default_model_params_shapes():
    dim = 1
    return [dim]

def get_inputs(batch_size=128, input_shape=4000, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_shape)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, input_shape, dtype=dtype)]
    else:
        return None

def get_model(dim=1, dtype=torch.float16):
    return Model(dim=dim, dtype=dtype)

def get_real_inputs(batch_size=128, input_shape=4000, dtype=torch.float16):
    return [torch.randn(batch_size, input_shape, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 128
    input_shape = 1000
    dim = 1
    range_batch_size = [1, 1024]
    range_input_shape = [1, 1023]
    range_dim = [0, 1]  # Exclusive cumulative sum can be along any dimension
    dtype = [int, int, int]
    return [batch_size, input_shape, dim], \
           [range_batch_size, range_input_shape, range_dim], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, input_shape, dim):
    return [batch_size, input_shape], [dim]
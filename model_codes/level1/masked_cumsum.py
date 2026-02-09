import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """

    def __init__(self, dim, dtype=torch.float16):
        super(Model, self).__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x, mask):
        return torch.cumsum(x * mask, dim=self.dim)

def get_default_input_shapes():
    batch_size = 128
    input_shape = 4000
    
    return [batch_size, input_shape]

def get_default_model_params_shapes():
    # No learnable parameters for masked_cumsum
    dim = 1
    return [dim]

def get_inputs(batch_size=128, input_shape=4000, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_shape*2)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, input_shape, dtype=dtype)
        mask = torch.randint(0, 2, (batch_size, input_shape)).bool()
        return [x, mask]
    else:
        return None

def get_model(dim=1, dtype=torch.float16):
    return Model(dim=dim, dtype=dtype)

def get_real_inputs(batch_size=128, input_shape=4000, dtype=torch.float16):
    x = torch.randn(batch_size, input_shape, dtype=dtype)
    mask = torch.randint(0, 2, (batch_size, input_shape)).bool()
    return [x, mask]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 128
    input_shape = 4000
    dim = 1
    range_batch_size = [1, 1024]
    range_input_shape = [1, 4096]
    range_dim = [0, 1]  # Masked cumulative sum can be along any dimension
    dtype = [int, int, int]
    
    return [batch_size, input_shape, dim], \
           [range_batch_size, range_input_shape, range_dim], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, input_shape, dim):
    return [batch_size, input_shape], [dim]
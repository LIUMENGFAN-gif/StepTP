import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def get_model(shape=None,dtype=torch.float16):
    return Model(dtype)

def get_default_input_shapes():
    batch_size = 1024
    dim = 1024
    return [batch_size, dim]

def get_default_model_params_shapes():
    # No model parameters for this model
    return []

def get_inputs(batch_size=2000, dim=2000, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*dim)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, dim, dtype=dtype)
        return [x]
    else:
        return None

def get_real_inputs(batch_size=2000, dim=2000, dtype=torch.float16):
    x = torch.randn(batch_size, dim, dtype=dtype)
    return [x]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 1024
    dim = 1024
    range_batch_size = [1, 1024]
    range_dim = [1, 10683*2]
    dtype = [int, int]
    return [batch_size, dim], \
           [range_batch_size, range_dim], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, dim):
    return [batch_size, dim], []  # No model parameters in this case
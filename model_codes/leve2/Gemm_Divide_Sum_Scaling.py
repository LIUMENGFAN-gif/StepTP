import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of output features.
        scaling_factor (float): Scaling factor for the output.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, input_size, hidden_size, scaling_factor, dtype=torch.float16):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dtype))
        self.scaling_factor = torch.tensor(scaling_factor, dtype=dtype)

    def forward(self, x):
        x = torch.matmul(x, self.weight.T)
        x = x / 2
        x = torch.sum(x, dim=1, keepdim=True)
        x = x * self.scaling_factor
        return x

def get_default_input_shapes():
    batch_size = 128
    input_size = 10
    return [batch_size, input_size]

def get_default_model_params_shapes():
    input_size = 10
    hidden_size = 20
    scaling_factor = 1.5
    return [input_size, hidden_size, scaling_factor]

def get_inputs(batch_size=128, input_size=10, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_size)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, input_size, dtype=dtype)]
    else:
        return None

def get_model(input_size=10, hidden_size=20, scaling_factor=1.5, dtype=torch.float16):
    return Model(input_size, hidden_size, scaling_factor, dtype=dtype)

def get_real_inputs(batch_size=128, input_size=10, dtype=torch.float16):
    return [torch.randn(batch_size, input_size, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    input_size = 10
    hidden_size = 20
    scaling_factor = 1.5
    batch_size = 128
    range_input_size = [1, 4096]
    range_hidden_size = [1, 1024]
    range_scaling_factor = [0.1, 10.0]
    range_batch_size = [1, 1024]
    dtype = [int, int, float, int]
    return [input_size, hidden_size, scaling_factor, batch_size], \
           [range_input_size, range_hidden_size, range_scaling_factor, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(input_size, hidden_size, scaling_factor, batch_size):
    return [batch_size, input_size], \
           [input_size, hidden_size, scaling_factor]
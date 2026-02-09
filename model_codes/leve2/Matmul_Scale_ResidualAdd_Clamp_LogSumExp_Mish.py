import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of output features.
        scale_factor (float): Scaling factor for the output.
        clamp_min (float): Minimum clamp value.
        clamp_max (float): Maximum clamp value.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max, dtype=torch.float16):
        super(Model, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size).to(dtype)
        self.scale_factor = torch.tensor(scale_factor, dtype=dtype)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.matmul(x)
        x = x * self.scale_factor
        x = x + x
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.nn.functional.mish(x)
        return x

def get_default_input_shapes():
    batch_size = 128
    input_size = 512
    return [batch_size, input_size]

def get_default_model_params_shapes():
    input_size = 512
    hidden_size = 1024
    scale_factor = 2.0
    clamp_min = -10.0
    clamp_max = 10.0
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]

def get_inputs(batch_size=128, input_size=512, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_size) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, input_size, dtype=dtype)]
    else:
        return None

def get_model(input_size=512, hidden_size=1024, scale_factor=2.0, clamp_min=-10.0, clamp_max=10.0, dtype=torch.float16):
    return Model(input_size, hidden_size, scale_factor, clamp_min, clamp_max, dtype=dtype)

def get_real_inputs(batch_size=128, input_size=512, dtype=torch.float16):
    return [torch.randn(batch_size, input_size, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    input_size = 512
    hidden_size = 1024
    scale_factor = 2.0
    clamp_min = -10.0
    clamp_max = 10.0
    batch_size = 128
    range_input_size = [64, 4096]
    range_hidden_size = [64, 4096]
    range_scale_factor = [0.1, 10.0]
    range_clamp_min = [-20.0, -1.0]
    range_clamp_max = [1.0, 20.0]
    range_batch_size = [1, 1024]
    dtype = [int, int, float, float, float, int]
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max, batch_size], \
           [range_input_size, range_hidden_size, range_scale_factor, range_clamp_min, range_clamp_max, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(input_size, hidden_size, scale_factor, clamp_min, clamp_max, batch_size):
    return [batch_size, input_size], \
           [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
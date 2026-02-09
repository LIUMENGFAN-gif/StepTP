import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """
    def __init__(self, input_size, output_size, divisor, dtype=torch.float16):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size).to(dtype)
        self.divisor = torch.tensor(divisor, dtype=dtype)

    def forward(self, x):
        x = self.linear(x)
        x = x / self.divisor
        x = torch.nn.functional.gelu(x)
        return x

def get_default_input_shapes():
    batch_size = 128
    input_size = 512
    return [batch_size, input_size]

def get_default_model_params_shapes():
    input_size = 512
    output_size = 1024
    divisor = 10.0
    return [input_size, output_size, divisor]

def get_inputs(batch_size=128, input_size=512, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_size) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, input_size, dtype=dtype)]
    else:
        return None

def get_model(input_size=512, output_size=1024, divisor=10.0, dtype=torch.float16):
    return Model(input_size, output_size, divisor, dtype=dtype)

def get_real_inputs(batch_size=128, input_size=512, dtype=torch.float16):
    return [torch.randn(batch_size, input_size, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    input_size = 512
    output_size = 1024
    divisor = 10.0
    batch_size = 128
    range_input_size = [8, 4096]
    range_output_size = [8, 4096]
    range_divisor = [1.0, 100.0]
    range_batch_size = [1, 1024]
    dtype = [int, int, float, int]
    return [input_size, output_size, divisor, batch_size], \
           [range_input_size, range_output_size, range_divisor, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(input_size, output_size, divisor, batch_size):
    return [batch_size, input_size], \
           [input_size, output_size, divisor]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """
    def __init__(self, input_size, hidden_size, dtype=torch.float16):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x

def get_default_input_shapes():
    batch_size = 128
    input_size = 10
    return [batch_size, input_size]

def get_default_model_params_shapes():
    input_size = 10
    hidden_size = 20
    return [input_size, hidden_size]

def get_inputs(batch_size, input_size, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_size) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, input_size, dtype=dtype)]
    else:
        return None

def get_model(input_size, hidden_size, dtype=torch.float16):
    return Model(input_size, hidden_size, dtype=dtype)

def get_real_inputs(batch_size=128, input_size=10, dtype=torch.float16):
    return [torch.randn(batch_size, input_size, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    input_size = 10
    hidden_size = 20
    batch_size = 128
    range_input_size = [8, 4096]
    range_hidden_size = [8, 4096]
    range_batch_size = [1, 1024]
    dtype = [int, int, int]
    return [input_size, hidden_size, batch_size], \
           [range_input_size, range_hidden_size, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(input_size, hidden_size, batch_size):
    return [batch_size, input_size], \
           [input_size, hidden_size]
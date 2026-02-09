import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies minimum, and subtracts a constant.
    """
    def __init__(self, in_features, out_features, constant, dtype=torch.float16):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features).to(dtype)
        self.constant = nn.Parameter(torch.tensor(constant, dtype=dtype))

    def forward(self, x):
        x = self.linear(x)
        x = torch.min(x, self.constant)
        x = x - self.constant
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 10
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features = 10
    out_features = 5
    constant = 2.0
    return [in_features, out_features, constant]

def get_inputs(batch_size, in_features, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_features, dtype=dtype)]
    else:
        return None

def get_model(in_features, out_features, constant, dtype=torch.float16):
    return Model(in_features, out_features, constant, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=10, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 10
    out_features = 5
    constant = 2.0
    batch_size = 128
    range_in_features = [1, 4096]
    range_out_features = [1, 4096]
    range_constant = [0.1, 10.0]
    range_batch_size = [1, 1024]
    dtype = [int, int, float, int]
    return [in_features, out_features, constant, batch_size], \
           [range_in_features, range_out_features, range_constant, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, constant, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features, constant]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features, dtype=torch.float16):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features).to(dtype)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 100
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features = 100
    out_features = 10
    return [in_features, out_features]

def get_inputs(batch_size=128, in_features=100, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_features, dtype=dtype)]
    else:
        return None

def get_model(in_features=100, out_features=10, dtype=torch.float16):
    return Model(in_features, out_features, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=100, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 100
    out_features = 10
    batch_size = 128
    range_in_features = [8, 4096]
    range_out_features = [8, 4096]
    range_batch_size = [1, 1024]
    dtype = [int, int, int]
    return [in_features, out_features, batch_size], \
           [range_in_features, range_out_features, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features]
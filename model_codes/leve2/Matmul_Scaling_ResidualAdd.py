import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs matrix multiplication, scaling, and residual addition.
    """
    def __init__(self, in_features, out_features, dtype=torch.float16):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features).to(dtype)
        self.scale = nn.Parameter(torch.randn(out_features, dtype=dtype))


    def forward(self, x, residual):
        x = self.linear(x)
        x = x * self.scale
        x = x + residual
        return x

def get_default_input_shapes():
    batch_size = 64
    in_features = 128
    out_features = 256
    # x: [batch_size, in_features], residual: [batch_size, out_features]
    return [batch_size, in_features, batch_size, out_features]

def get_default_model_params_shapes():
    in_features = 128
    out_features = 256
    # scale: [out_features]
    return [in_features, out_features]

def get_inputs(batch_size, in_features, batch_size2, out_features, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features+batch_size2*out_features) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [
            torch.randn(batch_size, in_features, dtype=dtype),
            torch.randn(batch_size2, out_features, dtype=dtype)
        ]
    else:
        return None

def get_model(in_features, out_features, dtype=torch.float16):
    return Model(in_features, out_features, dtype=dtype)

def get_real_inputs(batch_size=64, in_features=256, batch_size2=64, out_features=256, dtype=torch.float16):
    return [
        torch.randn(batch_size, in_features, dtype=dtype),
        torch.randn(batch_size2, out_features, dtype=dtype)
    ]

def set_default_shapes_ranges_and_dtypes():
    in_features = 128
    out_features = 256
    batch_size = 64
    range_in_features = [8, 4096]
    range_out_features = [8, 4096]
    range_batch_size = [1, 1024]
    dtype = [int, int, int]
    return [in_features, out_features, batch_size], \
           [range_in_features, range_out_features, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, batch_size):
    return [batch_size, in_features, batch_size, out_features], \
           [in_features, out_features]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float16):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias).to(dtype)

    def forward(self, x):
        x = self.gemm(x)
        x = x * torch.sigmoid(x) # Swish activation
        x = x / 2.0
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = torch.tanh(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 1024
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features = 1024
    out_features = 512
    return [in_features, out_features]

def get_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_features, dtype=dtype)]
    else:
        return None

def get_model(in_features=1024, out_features=512, bias=True, dtype=torch.float16):
    return Model(in_features, out_features, bias=bias, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 1024
    out_features = 512
    batch_size = 128
    range_in_features = [8, 2048]
    range_out_features = [8, 2048]
    range_batch_size = [1, 1024]
    dtype = [int, int, int]
    return [in_features, out_features, batch_size], \
           [range_in_features, range_out_features, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features]
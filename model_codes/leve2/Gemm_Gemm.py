import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, dtype=torch.float16):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features).to(dtype)

    def forward(self, x, y):
        a = self.gemm(x)
        b = self.gemm(y)
        return a+b

def get_default_input_shapes():
    batch_size = 128
    in_features = 1024
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features = 1024
    out_features = 512
    return [in_features, out_features]

def get_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    return [torch.empty(batch_size, in_features, dtype=dtype), torch.empty(batch_size, in_features, dtype=dtype)]

def get_model(in_features=1024, out_features=512, dtype=torch.float16):
    return Model(in_features, out_features, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype), torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 1024
    out_features = 512
    batch_size = 128
    range_in_features = [1, 4089]
    range_out_features = [1, 4089]
    range_batch_size = [1, 1024]
    dtype = [int, int, int]
    return [in_features, out_features, batch_size], \
           [range_in_features, range_out_features, range_batch_size], \
           dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features]
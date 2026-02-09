import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    Parameters:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        multiplier (float): Scalar multiplier for the output.
        negative_slope (float): Negative slope for LeakyReLU.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope, dtype=torch.float16):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, dtype=dtype)
        self.multiplier = torch.tensor(multiplier,dtype=dtype)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dtype = dtype

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 1024
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features=1024 
    out_features=512
    multiplier = 2.0
    negative_slope=0.1 
    return [in_features, out_features, multiplier, negative_slope]

def get_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_features, dtype=dtype)]
    else:
        return None

def get_model(in_features=1024, out_features=512, multiplier=2.0, negative_slope=0.1, dtype=torch.float16):
    return Model(in_features, out_features, multiplier, negative_slope, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 1024
    out_features = 512
    multiplier = 2.0
    negative_slope = 0.1
    batch_size = 128
    range_in_features = [64, 2048]
    range_out_features = [64, 2048]
    range_multiplier = [0.1, 10.0]
    range_negative_slope = [0.01, 0.2]
    range_batch_size = [1, 1024]
    dtype = [int, int, float, float, int]
    return [in_features, out_features, multiplier, negative_slope, batch_size], \
           [range_in_features, range_out_features, range_multiplier, range_negative_slope, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, multiplier, negative_slope, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features, multiplier, negative_slope]
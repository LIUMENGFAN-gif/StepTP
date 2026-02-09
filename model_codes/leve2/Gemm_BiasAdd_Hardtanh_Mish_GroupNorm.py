import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups, dtype=torch.float16):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features).to(dtype)
        self.bias = nn.Parameter(torch.randn(*bias_shape, dtype=dtype))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features).to(dtype)

    def forward(self, x):
        x = self.gemm(x)
        x = x + self.bias
        x = self.hardtanh(x)
        x = self.mish(x)
        x = self.groupnorm(x)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 512
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features = 512
    out_features = 1024
    bias_shape = [1024]
    num_groups = 32
    return [in_features, out_features, bias_shape, num_groups]

def get_inputs(batch_size=128, in_features=512, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_features, dtype=dtype)]
    else:
        return None

def get_model(in_features=512, out_features=1024, bias_shape=[1024], num_groups=32, dtype=torch.float16):
    return Model(in_features, out_features, bias_shape, num_groups, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=512, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 512
    out_features = 1024
    num_groups = 32
    batch_size = 128
    range_in_features = [64, 2048]
    range_out_features = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    range_num_groups = [2, 4, 8]
    range_batch_size = [1, 1024]
    dtype = [int, int, int, int]
    return [in_features, out_features, num_groups, batch_size], \
           [range_in_features, range_out_features, range_num_groups, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, num_groups, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features, [out_features], num_groups]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max, dtype=torch.float16):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features).to(dtype)
        self.scaling_factor = torch.tensor(scaling_factor, dtype=dtype)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max).to(dtype)
        self.gelu = nn.GELU().to(dtype)
        self.dtype = dtype

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scaling_factor
        x = self.hardtanh(x)
        x = self.gelu(x)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 1024
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features = 1024
    out_features = 512
    scaling_factor = 0.5
    hardtanh_min = -2
    hardtanh_max = 2
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]

def get_inputs(batch_size, in_features, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_features, dtype=dtype)]
    else:
        return None

def get_model(in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max, dtype=torch.float16):
    return Model(in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 1024
    out_features = 512
    scaling_factor = 0.5
    hardtanh_min = -2.0
    hardtanh_max = 2.0
    batch_size = 128
    range_in_features = [8, 2048]
    range_out_features = [8, 2048]
    range_scaling_factor = [0.1, 1.0]
    range_hardtanh_min = [-10.0, -5.0]
    range_hardtanh_max = [5.0, 10.0]
    range_batch_size = [1, 1024]
    dtype = [int, int, float, float, float, int]
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max, batch_size], \
           [range_in_features, range_out_features, range_scaling_factor, range_hardtanh_min, range_hardtanh_max, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]
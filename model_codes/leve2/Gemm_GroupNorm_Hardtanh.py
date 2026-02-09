import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    Parameters:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        num_groups (int): Number of groups for GroupNorm.
        hardtanh_min (float): Minimum value for HardTanh.
        hardtanh_max (float): Maximum value for HardTanh.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max, dtype=torch.float16):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features).to(dtype)
        self.group_norm = nn.GroupNorm(num_groups, out_features).to(dtype)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max).to(dtype)

    def forward(self, x):
        x = self.gemm(x)
        x=x.unsqueeze(-1).unsqueeze(-1)
        x = torch.cat((x,x),dim=-1)
        x = self.group_norm(x)
        x = self.hardtanh(x)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 1024
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features = 1024
    out_features = 512
    num_groups = 8
    hardtanh_min = -2.0
    hardtanh_max = 2.0
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]

def get_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_features, dtype=dtype)]
    else:
        return None

def get_model(in_features=1024, out_features=512, num_groups=8, hardtanh_min=-2.0, hardtanh_max=2.0, dtype=torch.float16):
    return Model(in_features, out_features, num_groups, hardtanh_min, hardtanh_max, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 1024
    out_features = 512
    batch_size = 128
    num_groups = 8
    hardtanh_min = -2.0
    hardtanh_max = 2.0
    range_in_features = [8, 2048]
    range_out_features = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    range_num_groups = [2, 4, 8]
    range_hardtanh_min = [-10.0, -5.0]
    range_hardtanh_max = [5.0, 10.0]
    range_batch_size = [1, 1024]
    dtype = [int, int, int, float, float, int]
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max, batch_size], \
           [range_in_features, range_out_features, range_num_groups, range_hardtanh_min, range_hardtanh_max, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, num_groups, hardtanh_min, hardtanh_max, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]
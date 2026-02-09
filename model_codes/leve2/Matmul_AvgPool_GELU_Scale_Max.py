import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor, dtype=torch.float16):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features).to(dtype)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size).to(dtype)
        self.scale_factor = torch.tensor(scale_factor, dtype=dtype)

    def forward(self, x):
        x = self.matmul(x)
        x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        x = torch.nn.functional.gelu(x)
        x = x * self.scale_factor
        x = torch.max(x, dim=1)[0]
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 512
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features = 512
    out_features = 256
    pool_kernel_size = 4
    scale_factor = 2.0
    return [in_features, out_features, pool_kernel_size, scale_factor]

def get_inputs(batch_size=128, in_features=512, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_features, dtype=dtype)]
    else:
        return None

def get_model(in_features=512, out_features=256, pool_kernel_size=4, scale_factor=2.0, dtype=torch.float16):
    return Model(in_features, out_features, pool_kernel_size, scale_factor, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=512, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 512
    out_features = 256
    pool_kernel_size = 4
    scale_factor = 2.0
    batch_size = 128
    range_in_features = [8, 4096]
    range_out_features = [8, 4096]
    range_pool_kernel_size = [2, 8]
    range_scale_factor = [0.1, 5.0]
    range_batch_size = [1, 1024]
    dtype = [int, int, int, float, int]
    return [in_features, out_features, pool_kernel_size, scale_factor, batch_size], \
           [range_in_features, range_out_features, range_pool_kernel_size, range_scale_factor, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, pool_kernel_size, scale_factor, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features, pool_kernel_size, scale_factor]
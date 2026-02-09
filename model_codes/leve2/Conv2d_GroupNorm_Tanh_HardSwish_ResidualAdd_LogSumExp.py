import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish,
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps).to(dtype)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        x_conv = self.conv(x)
        x_norm = self.group_norm(x_conv)
        x_tanh = self.tanh(x_norm)
        x_hard_swish = self.hard_swish(x_tanh)
        x_res = x_conv + x_hard_swish
        x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp

def get_default_input_shapes():
    batch_size = 128
    in_channels = 3
    height, width = 32, 32
    return [batch_size, in_channels, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    groups = 8
    return [in_channels, out_channels, kernel_size, groups]

def get_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=3, groups=8, eps=1e-5, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, groups, eps=eps, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    groups = 8
    eps = 1e-5
    batch_size = 128
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [8,16,64,128]
    range_kernel_size = [1, 7]
    range_groups = [2,4,8]
    range_eps = [1e-6, 1e-4]
    range_batch_size = [1, 256]
    range_height = [1, 256]
    range_width = [1, 256]
    dtypes = [int, int, int, int, float, int, int, int]
    return [in_channels, out_channels, kernel_size, groups, eps, batch_size, height, width], \
           [range_in_channels, range_out_channels, range_kernel_size,
            range_groups, range_eps, range_batch_size,
            range_height, range_width], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, groups, eps, batch_size, height, width):
    return [batch_size, in_channels, height, width], [in_channels, out_channels, kernel_size, groups, eps]
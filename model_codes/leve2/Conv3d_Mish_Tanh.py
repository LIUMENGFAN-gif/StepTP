import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False).to(dtype)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.mish(x)
        x = torch.tanh(x)
        return x

def get_default_input_shapes():
    batch_size = 64
    in_channels = 3
    D = 16
    H = 32
    W = 32
    return [batch_size, in_channels, D, H, W]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 1
    padding = 0
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs(batch_size, in_channels, D, H, W, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*D*H*W)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, D, H, W, dtype=dtype)]
    else:
        return None

def get_model(in_channels, out_channels, kernel_size, stride=1, padding=0, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dtype=dtype)

def get_real_inputs(batch_size=16, in_channels=3, D=16, H=32, W=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, D, H, W, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 1
    padding = 0
    batch_size = 64
    D = 16
    H = 32
    W = 32
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size = [1, 5]
    range_stride = [1, 4]
    range_padding = [0, 3]
    range_batch_size = [1, 128]
    range_D = [1, 64]
    range_H = [1, 64]
    range_W = [1, 64]
    dtypes = [int, int, int, int, int, int, int, int, int]
    return [in_channels, out_channels, kernel_size, stride, padding, batch_size, D, H, W], \
           [range_in_channels, range_out_channels, range_kernel_size, range_stride, range_padding,
            range_batch_size, range_D, range_H, range_W], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, stride, padding, batch_size, D, H, W):
    return [batch_size, in_channels, D, H, W], \
           [in_channels, out_channels, kernel_size, stride, padding]
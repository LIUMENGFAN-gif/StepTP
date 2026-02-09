import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        scale_factor (float): Scaling factor for the output.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, dtype=torch.float16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.scale_factor = torch.tensor(scale_factor, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        x = x * self.scale_factor
        x = torch.min(x, dim=1, keepdim=True)[0]
        return x

def get_default_input_shapes():
    batch_size = 128
    in_channels = 3
    height = 32
    width = 32
    return [batch_size, in_channels, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    scale_factor = 2.0
    return [in_channels, out_channels, kernel_size, scale_factor]

def get_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=3, scale_factor=2.0, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, scale_factor, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    scale_factor = 2.0
    batch_size = 128
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size = [1, 7]
    range_scale_factor = [0.1, 10.0]
    range_batch_size = [1, 256]
    range_height = [1, 256]
    range_width = [1, 256]
    dtypes = [int, int, int, float, int, int, int]
    return [in_channels, out_channels, kernel_size, scale_factor, batch_size, height, width],\
            [range_in_channels, range_out_channels, range_kernel_size,
             range_scale_factor, range_batch_size, range_height, range_width], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, scale_factor, batch_size, height, width):
    return [batch_size, in_channels, height, width], [in_channels, out_channels, kernel_size, scale_factor]
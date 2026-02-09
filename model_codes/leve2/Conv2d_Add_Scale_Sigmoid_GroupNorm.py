import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        num_groups (int): Number of groups for GroupNorm.
        bias_shape (tuple): Shape of the bias parameter.
        scale_shape (tuple): Shape of the scale parameter.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, dtype=dtype)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=dtype))
        self.scale = nn.Parameter(torch.randn(scale_shape, dtype=dtype))
        self.group_norm = nn.GroupNorm(num_groups, out_channels, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        x = x + self.bias
        x = x * self.scale
        x = torch.sigmoid(x)
        x = self.group_norm(x)
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
    num_groups = 8
    bias_shape = (out_channels, 1, 1)
    scale_shape = (out_channels, 1, 1)
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]

def get_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=3, num_groups=8, bias_shape=(16, 1, 1), scale_shape=(16, 1, 1), dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    num_groups = 8
    batch_size = 128
    height = 32
    width = 32
    range_in_channels = [1,64]
    range_out_channels = [8,16,64]
    range_kernel_size = [1, 7]
    range_num_groups = [2,4,8]
    range_batch_size = [1, 256]
    range_height = [1, 256]
    range_width = [1, 256]
    dtypes = [int, int, int, int, int, int, int]
    return [in_channels, out_channels, kernel_size, num_groups, batch_size, height, width],[range_in_channels, range_out_channels, range_kernel_size, range_num_groups,range_batch_size, range_height, range_width], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, num_groups, batch_size, height, width):
    return [batch_size, in_channels, height, width], [in_channels, out_channels, kernel_size, num_groups, (out_channels, 1, 1), (out_channels, 1, 1)]
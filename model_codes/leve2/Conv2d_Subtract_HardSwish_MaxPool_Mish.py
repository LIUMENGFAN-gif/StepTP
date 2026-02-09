import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        subtract_value (float): Value to subtract after convolution.
        pool_kernel_size (int): Kernel size for MaxPool2d.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size, dtype=torch.float16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.subtract_value = torch.tensor(subtract_value, dtype=dtype)
        self.pool = nn.MaxPool2d(pool_kernel_size).to(dtype)

    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract_value
        x = torch.nn.functional.hardswish(x)
        x = self.pool(x)
        x = torch.nn.functional.mish(x)
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
    subtract_value = 0.5
    pool_kernel_size = 2
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]

def get_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=3, subtract_value=0.5, pool_kernel_size=2, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    subtract_value = 0.5
    pool_kernel_size = 2
    batch_size = 128
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size = [1, 7]
    range_subtract_value = [0.0, 1.0]
    range_pool_kernel_size = [1, 5]
    range_batch_size = [1, 256]
    range_height = [1, 256]
    range_width = [1, 256]
    dtypes = [int, int, int, float, int, int, int, int]
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size, batch_size, height, width], \
           [range_in_channels, range_out_channels, range_kernel_size,
            range_subtract_value, range_pool_kernel_size, range_batch_size, range_height, range_width], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size, batch_size, height, width):
    return [batch_size, in_channels, height, width], [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]
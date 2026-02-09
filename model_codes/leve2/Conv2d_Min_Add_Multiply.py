import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 2D convolution, min with constant, bias add, and scaling.
    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        constant_value (float): Constant for min operation.
        bias_shape (tuple): Shape of the bias parameter.
        scaling_factor (float): Scaling factor for the output.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor, dtype=torch.float16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.constant_value = torch.tensor(constant_value, dtype=dtype)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=dtype))
        self.scaling_factor = torch.tensor(scaling_factor, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        x = torch.min(x, self.constant_value)
        x = x + self.bias
        x = x * self.scaling_factor
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
    constant_value = 0.5
    bias_shape = (out_channels, 1, 1)
    scaling_factor = 2.0
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]

def get_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=3, constant_value=0.5, bias_shape=(16, 1, 1), scaling_factor=2.0, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    constant_value = 0.5
    scaling_factor = 2.0
    batch_size = 128
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size = [1, 7]
    range_constant_value = [0.0, 1.0]
    range_scaling_factor = [0.1, 10.0]
    range_batch_size = [1, 256]
    range_height = [1, 256]
    range_width = [1, 256]
    dtypes = [int, int, int, float, float, int, int, int]
    return [in_channels, out_channels, kernel_size, constant_value, scaling_factor,
            batch_size, height, width], \
           [range_in_channels, range_out_channels, range_kernel_size,
            range_constant_value, range_scaling_factor,
            range_batch_size, range_height, range_width], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, constant_value, scaling_factor, batch_size, height, width):
    return [batch_size, in_channels, height, width], [in_channels, out_channels, kernel_size, constant_value, (out_channels, 1, 1), scaling_factor]
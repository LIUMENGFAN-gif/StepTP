import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.scaling_factor = torch.tensor(scaling_factor, dtype=dtype)
        self.bias = nn.Parameter(torch.randn(*bias_shape, dtype=dtype))
        self.max_pool = nn.MaxPool2d(pool_kernel_size).to(dtype)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = x * self.scaling_factor
        x = x + self.bias
        x = self.max_pool(x)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_channels = 3
    height, width = 32, 32
    return [batch_size, in_channels, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    scaling_factor = 2.0
    bias_shape = [out_channels, 1, 1]
    pool_kernel_size = 2
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]

def get_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=3, scaling_factor=2.0, bias_shape=[16, 1, 1], pool_kernel_size=2, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    scaling_factor = 2.0
    pool_kernel_size = 2
    batch_size = 128
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size = [1, 7]
    range_scaling_factor = [0.1, 10.0]
    range_pool_kernel_size = [1, 5]
    range_batch_size = [1, 256]
    range_height = [1, 256]
    range_width = [1, 256]
    dtypes = [int, int, int, float, int, int, int, int]
    
    return [in_channels, out_channels, kernel_size, scaling_factor, pool_kernel_size,
             batch_size, height, width],\
            [range_in_channels, range_out_channels, range_kernel_size,
             range_scaling_factor,
             range_pool_kernel_size,
             range_batch_size, range_height, range_width], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, scaling_factor, pool_kernel_size, batch_size, height, width):
    return [batch_size, in_channels, height, width], \
           [in_channels, out_channels, kernel_size, scaling_factor, [out_channels, 1, 1], pool_kernel_size]
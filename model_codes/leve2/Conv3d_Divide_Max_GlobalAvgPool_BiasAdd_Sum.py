import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.divisor = torch.tensor(divisor, dtype=dtype)
        self.max_pool = nn.MaxPool3d(pool_size).to(dtype)
        # self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(*bias_shape, dtype=dtype))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = self.max_pool(x)
        # x = self.global_avg_pool(x)
        x = torch.mean(x, dim=(-1, -2, -3), keepdim=True)  # Global average pooling
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 32, 32
    return [batch_size, in_channels, depth, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 16
    kernel_size = [3, 3, 3]
    divisor = 2.0
    pool_size = 2
    bias_shape = [16, 1, 1, 1]
    sum_dim = 1
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs(batch_size=128, in_channels=3, depth=16, height=32, width=32, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*depth*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, depth, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=[3, 3, 3], divisor=2.0, pool_size=[2, 2, 2], bias_shape=[16, 1, 1, 1], sum_dim=1, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, depth=16, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, depth, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    divisor = 2.0
    pool_size = 2
    batch_size = 128
    depth = 16
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size = [1, 5]
    range_divisor = [0.1, 10.0]
    range_pool_size = [1, 5]
    range_batch_size = [1, 128]
    range_depth = [1, 64]
    range_height = [1, 64]
    range_width = [1, 64]
    dtypes = [int, int, int, float, int, int, int, int, int]
    return [in_channels, out_channels, kernel_size, divisor, pool_size,
             batch_size, depth, height, width],\
            [range_in_channels, range_out_channels, range_kernel_size,
             range_divisor,
             range_pool_size,
             range_batch_size, range_depth, range_height, range_width], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, divisor, pool_size, batch_size, depth, height, width):
    return [batch_size, in_channels, depth, height, width], \
           [in_channels, out_channels, [kernel_size,kernel_size,kernel_size], divisor, pool_size, [out_channels, 1, 1, 1], 1]

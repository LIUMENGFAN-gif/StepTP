import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.group_norm = nn.GroupNorm(num_groups, out_channels).to(dtype)
        self.scale = nn.Parameter(torch.ones(*scale_shape, dtype=dtype))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size).to(dtype)
        self.clamp_min = torch.tensor(clamp_min, dtype=dtype)
        self.clamp_max = torch.tensor(clamp_max, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = torch.clamp(x, self.clamp_min.item(), self.clamp_max.item())
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
    num_groups = 8
    scale_shape = [16, 1, 1]
    maxpool_kernel_size = 2
    clamp_min = 0.0
    clamp_max = 1.0
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]

def get_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=3, num_groups=8, scale_shape=[16, 1, 1], maxpool_kernel_size=2, clamp_min=0.0, clamp_max=1.0, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    num_groups = 8
    scale_shape1 = 16
    maxpool_kernel_size = 2
    clamp_min = 0.0
    clamp_max = 1.0
    batch_size = 128
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [8,16,64,128]
    range_kernel_size = [1, 7]
    range_num_groups = [2,4,8]
    range_scale_shape1 = [1, 64]
    range_maxpool_kernel_size = [1, 7]
    range_clamp_min = [-10.0, 10.0]
    range_clamp_max = [-10.0, 10.0]
    range_batch_size = [1, 256]
    range_height = [1, 256]
    range_width = [1, 256]
    dtypes = [int, int, int, int, int, int, float, float, int, int, int]
    return [in_channels, out_channels, kernel_size, num_groups,
             scale_shape1, maxpool_kernel_size, clamp_min, clamp_max,
             batch_size, height, width], [range_in_channels, range_out_channels,
             range_kernel_size, range_num_groups,
             range_scale_shape1,
             range_maxpool_kernel_size,
             range_clamp_min,
             range_clamp_max,
             range_batch_size,
             range_height,
             range_width], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, num_groups, scale_shape1, maxpool_kernel_size, clamp_min, clamp_max, batch_size, height, width):
    return [batch_size, in_channels, height, width], [in_channels, out_channels, kernel_size, num_groups, [scale_shape1,1,1], maxpool_kernel_size, clamp_min, clamp_max]
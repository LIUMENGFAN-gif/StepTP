import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.subtract1_value = torch.tensor(subtract1_value, dtype=dtype)
        self.subtract2_value = torch.tensor(subtract2_value, dtype=dtype)
        self.avgpool = nn.AvgPool2d(kernel_size_pool).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
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
    subtract1_value = 0.5
    subtract2_value = 0.2
    kernel_size_pool = 2
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]

def get_inputs(batch_size, in_channels, height, width, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    subtract1_value = 0.5
    subtract2_value = 0.2
    kernel_size_pool = 2
    batch_size = 128
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size = [1, 7]
    range_subtract1_value = [0.0, 1.0]
    range_subtract2_value = [0.0, 1.0]
    range_kernel_size_pool = [1, 5]
    range_batch_size = [1, 256]
    range_height = [1, 256]
    range_width = [1, 256]
    dtypes = [int, int, int, float, float, int, int, int,int]
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool,
             batch_size, height, width],\
            [range_in_channels, range_out_channels, range_kernel_size,
             range_subtract1_value, range_subtract2_value,
             range_kernel_size_pool,
             range_batch_size, range_height, range_width], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool, batch_size, height, width):
    return [batch_size, in_channels, height, width], \
            [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
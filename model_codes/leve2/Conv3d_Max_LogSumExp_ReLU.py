import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False).to(dtype)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2).to(dtype)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.relu(x)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_channels = 3
    depth = 16
    height = 32
    width = 32
    return [batch_size, in_channels, depth, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 1
    padding = 1
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs(batch_size, in_channels, depth, height, width, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*depth*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, depth, height, width, dtype=dtype)]
    else:
        return None

def get_model(in_channels, out_channels, kernel_size, stride, padding, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, stride, padding, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, depth=16, height=32, width=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, depth, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 1
    padding = 1
    batch_size = 128
    depth = 16
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size = [1, 5]
    range_stride = [1, 4]
    range_padding = [0, 3]
    range_batch_size = [1, 128]
    range_depth = [1, 64]
    range_height = [1, 64]
    range_width = [1, 64]
    dtype=[int, int, int, int, int, int, int, int, int]
    return [in_channels, out_channels, kernel_size, stride, padding, batch_size, depth, height, width], \
           [range_in_channels, range_out_channels, range_kernel_size, range_stride, range_padding,
            range_batch_size, range_depth, range_height, range_width], dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, stride, padding, batch_size, depth, height, width):
    return [batch_size, in_channels, depth, height, width], \
           [in_channels, out_channels, kernel_size, stride, padding]
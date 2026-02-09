import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        dtype (torch.dtype, optional): Data type for the weights and input. Defaults to torch.float16.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.conv3d = self.conv3d.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d(x)

def get_model(in_channels=3, out_channels=64, kernel_size=(3, 5, 7), stride=1, padding=0, dilation=1, groups=1, bias=False, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype)

def get_default_input_shapes():
    batch_size = 16
    in_channels = 3
    width = 64
    height = 64
    depth = 64
    return [batch_size, in_channels, width, height, depth]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    return [in_channels, out_channels, kernel_size]

def get_inputs(batch_size=16, in_channels=3, width=64, height=64, depth=64, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, width, height, depth, dtype=dtype, device='meta')
    return [x]

def get_real_inputs(batch_size=16, in_channels=3, width=64, height=64, depth=64, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, width, height, depth, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 64
    kernel_size_w = 3
    kernel_size_h = 5
    kernel_size_d = 7
    stride = 1
    padding = 0
    dilation = 1
    batch_size = 16
    width = 64
    height = 64
    depth = 64
    range_in_channels = [1, 256]
    range_out_channels = [1, 256]
    range_kernel_size_w = [1, 8]
    range_kernel_size_h = [1, 8]
    range_kernel_size_d = [1, 8]
    range_stride = [1, 4]
    range_padding = [0, 3]
    range_dilation = [1, 4]
    range_batch_size = [1, 1024]
    range_width = [8, 2048]
    range_height = [8, 2048]
    range_depth = [8, 2048]
    dtype=[int]*12
    return [in_channels, out_channels, kernel_size_w, kernel_size_h, kernel_size_d, stride, padding, dilation, batch_size, width, height, depth], \
           [range_in_channels, range_out_channels, range_kernel_size_w, range_kernel_size_h, range_kernel_size_d, range_stride, range_padding, range_dilation, range_batch_size, range_width, range_height, range_depth], dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size_w, kernel_size_h, kernel_size_d, stride, padding, dilation, batch_size, width, height, depth):
    return [batch_size, in_channels, width, height, depth], \
           [in_channels, out_channels, (kernel_size_w, kernel_size_h, kernel_size_d), stride, padding, dilation]
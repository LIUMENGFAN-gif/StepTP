import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.conv3d(x)

def get_inputs(batch_size, in_channels, depth, width, height, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, depth, width, height, dtype=dtype, device='meta')
    return [x]

def get_default_input_shapes():
    batch_size = 16
    in_channels = 3
    depth = 64
    width = 64
    height = 64
    return [batch_size, in_channels, depth, width, height]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]

def get_model(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=False, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype=dtype)

def get_real_inputs(batch_size=16, in_channels=3, depth=64, width=64, height=64, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, depth, width, height, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    batch_size = 16
    depth = 64
    width = 64
    height = 64
    range_in_channels = [1, 256]
    range_out_channels = [1, 256]
    range_kernel_size = [1, 8]
    range_stride = [1, 4]
    range_padding = [0, 3]
    range_dilation = [1, 4]
    range_batch_size = [1, 1024]
    range_depth = [8, 2048]
    range_width = [8, 2048]
    range_height = [8, 2048]
    dtype = [int] * 10
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, batch_size, depth, width, height], \
           [range_in_channels, range_out_channels, range_kernel_size, range_stride, range_padding, range_dilation, range_batch_size, range_depth, range_width, range_height], dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, stride, padding, dilation, batch_size, depth, width, height):
    return [batch_size, in_channels, depth, width, height], \
           [in_channels, out_channels, kernel_size, stride, padding, dilation]
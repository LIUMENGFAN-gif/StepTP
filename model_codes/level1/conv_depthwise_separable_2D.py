import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        dtype (torch.dtype, optional): Data type for the weights and input. Defaults to torch.float16.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.depthwise = self.depthwise.to(dtype)
        self.pointwise = self.pointwise.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def get_model(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, dtype)

def get_default_input_shapes():
    batch_size = 16
    in_channels = 3
    height = 256
    width = 256
    return [batch_size, in_channels, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]

def get_inputs(batch_size=16, in_channels=3, height=256, width=256, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, height, width, dtype=dtype, device='meta')
    return [x]

def get_real_inputs(batch_size=16, in_channels=3, height=256, width=256, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    batch_size = 16
    height = 256
    width = 256
    range_in_channels = [1, 512]
    range_out_channels = [1, 512]
    range_kernel_size = [1, 7]
    range_stride = [1, 4]
    range_padding = [0, 3]
    range_dilation = [1, 2]
    range_batch_size = [8, 1024]
    range_height = [8, 2048]
    range_width = [8, 2048]
    dtype = [int, int, int, int, int, int, int, int, int]
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, batch_size, height, width], \
           [range_in_channels, range_out_channels, range_kernel_size, range_stride, range_padding, range_dilation, range_batch_size, range_height, range_width], \
           dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, stride, padding, dilation, batch_size, height, width):
    return [batch_size, in_channels, height, width], \
           [in_channels, out_channels, kernel_size, stride, padding, dilation]
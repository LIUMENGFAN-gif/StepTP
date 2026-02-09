import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (tuple, optional): Padding applied to the input (top/bottom, left/right). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        dtype (torch.dtype, optional): Data type for the weights and input. Defaults to torch.float16.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.conv2d = self.conv2d.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)

def get_model(in_channels=3, out_channels=64, kernel_size=(3, 5), stride=1, padding=(1, 2), dilation=(2, 1), bias=False, dtype=torch.float16):
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
    kernel_size = (3, 5)
    stride = 1
    padding = (1, 2)
    dilation = (2, 1)
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]

def get_inputs(batch_size=16, in_channels=3, height=256, width=256, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, height, width, dtype=dtype, device='meta')
    return [x]

def get_real_inputs(batch_size=16, in_channels=3, height=256, width=256, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 64
    kernel_size_w = 3
    kernel_size_h = 5
    stride = 1
    padding_w = 1
    padding_h = 2
    dilation_w = 2
    dilation_h = 1
    batch_size = 16
    height = 256
    width = 256
    range_in_channels = [1, 512]
    range_out_channels = [1, 512]
    range_kernel_size_w = [1, 8]
    range_kernel_size_h = [1, 8]
    range_stride = [1, 4]
    range_padding_w = [0, 3]
    range_padding_h = [0, 3]
    range_dilation_w = [1, 2]
    range_dilation_h = [1, 2]
    range_batch_size = [1, 1024]
    range_height = [8, 2048]
    range_width = [8, 2048]
    dtype=[int,int,int,int,int,int,int,int,int,int,int,int]
    return [in_channels, out_channels, kernel_size_w, kernel_size_h, stride, padding_w, padding_h, dilation_w, dilation_h, batch_size, height, width], \
           [range_in_channels, range_out_channels, range_kernel_size_w, range_kernel_size_h,
            range_stride, range_padding_w, range_padding_h,
            range_dilation_w, range_dilation_h, range_batch_size, range_height, range_width], \
           dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size_w, kernel_size_h, stride, padding_w, padding_h, dilation_w, dilation_h, batch_size, height, width):
    return [batch_size, in_channels, height, width], \
           [in_channels, out_channels, (kernel_size_h, kernel_size_w), stride, (padding_h, padding_w), (dilation_h, dilation_w)]

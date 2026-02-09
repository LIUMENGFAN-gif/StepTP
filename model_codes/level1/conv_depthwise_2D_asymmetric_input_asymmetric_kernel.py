import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to in_channels.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        dtype (torch.dtype, optional): Data type for the weights and input. Defaults to torch.float16.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = None, bias: bool = False, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        if groups is None:
            groups = in_channels
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w), padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w), groups=groups, bias=bias
        )
        self.conv2d = self.conv2d.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)

def get_model(in_channels=3, out_channels=3, kernel_size_h=3, kernel_size_w=5, stride_h=1, stride_w=1, padding_h=0, padding_w=0, dilation_h=1, dilation_w=1, groups=None, bias=False, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, bias, dtype)

def get_default_input_shapes():
    batch_size = 16
    in_channels = 3
    height = 128
    width = 256
    return [batch_size, in_channels, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 3
    kernel_size_h = 3
    kernel_size_w = 5
    stride_h = 1
    stride_w = 1
    padding_h = 0
    padding_w = 0
    dilation_h = 1
    dilation_w = 1
    groups = in_channels
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]

def get_inputs(batch_size=16, in_channels=3, height=128, width=256, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, height, width, dtype=dtype, device='meta')
    return [x]

def get_real_inputs(batch_size=16, in_channels=3, height=128, width=256, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 3
    kernel_size_h = 3
    kernel_size_w = 5
    stride_h = 1
    stride_w = 1
    padding_h = 0
    padding_w = 0
    dilation_h = 1
    dilation_w = 1
    batch_size = 16
    height = 128
    width = 256
    range_in_channels = [1, 512]
    range_out_channels = [1, 512]
    range_kernel_size_h = [1, 7]
    range_kernel_size_w = [1, 7]
    range_stride_h = [1, 4]
    range_stride_w = [1, 4]
    range_padding_h = [0, 2]
    range_padding_w = [0, 2]
    range_dilation_h = [1, 2]
    range_dilation_w = [1, 2]
    range_batch_size = [2, 2048]
    range_height = [1, 512]
    range_width = [1, 512]
    dtype = [int] * 13  # All parameters are integers except for batch_size
    
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, batch_size, height, width], \
           [range_in_channels, range_out_channels, range_kernel_size_h, range_kernel_size_w, range_stride_h, range_stride_w, range_padding_h, range_padding_w, range_dilation_h, range_dilation_w, range_batch_size, range_height, range_width], \
           dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, batch_size, height, width):
    return [batch_size, in_channels, height, width], \
           [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w]  # Model parameters are convolution parameters
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        dtype (torch.dtype, optional): Data type for the weights and input. Defaults to torch.float16.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.conv3d = self.conv3d.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d(x)

def get_model(in_channels=3, out_channels=64, kernel_size=(3, 5, 7), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype)

def get_default_input_shapes():
    batch_size = 16
    in_channels = 3
    depth = 16
    height = 256
    width = 256
    return [batch_size, in_channels, depth, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    return [in_channels, out_channels, kernel_size]

def get_inputs(batch_size=16, in_channels=3, depth=16, height=256, width=256, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, depth, height, width, dtype=dtype, device='meta')
    return [x]

def get_real_inputs(batch_size=16, in_channels=3, depth=16, height=256, width=256, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, depth, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 64
    kernel_size_d = 3
    kernel_size_h = 5
    kernel_size_w = 7
    stride_d = 1
    stride_h = 1
    stride_w = 1
    padding_d = 0
    padding_h = 0
    padding_w = 0
    dilation_d = 1
    dilation_h = 1
    dilation_w = 1
    batch_size = 16
    depth = 16
    height = 256
    width = 256
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size_d = [1, 8]
    range_kernel_size_h = [1, 8]
    range_kernel_size_w = [1, 8]
    range_stride_d = [1, 2]
    range_stride_h = [1, 2]
    range_stride_w = [1, 2]
    range_padding_d = [0, 3]
    range_padding_h = [0, 3]
    range_padding_w = [0, 3]
    range_dilation_d = [1, 2]
    range_dilation_h = [1, 2]
    range_dilation_w = [1, 2]
    range_batch_size = [1, 1024]
    range_depth = [8, 1024]
    range_height = [8, 1024]
    range_width = [8, 1024]
    dtype=[int]*18
    return [in_channels, out_channels, kernel_size_d, kernel_size_h, kernel_size_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w,
            batch_size, depth, height, width], \
           [range_in_channels, range_out_channels,
            range_kernel_size_d, range_kernel_size_h, range_kernel_size_w,
            range_stride_d, range_stride_h, range_stride_w,
            range_padding_d, range_padding_h, range_padding_w,
            range_dilation_d, range_dilation_h, range_dilation_w,
            range_batch_size, range_depth, range_height, range_width], \
           dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size_d, kernel_size_h, kernel_size_w,
                                                    stride_d, stride_h, stride_w,
                                                    padding_d, padding_h, padding_w,
                                                    dilation_d, dilation_h, dilation_w,
                                                    batch_size, depth, height, width):
    return [batch_size, in_channels, depth, height, width], \
           [in_channels, out_channels, (kernel_size_d, kernel_size_h, kernel_size_w),
            (stride_d, stride_h, stride_w),
            (padding_d, padding_h, padding_w),
            (dilation_d, dilation_h, dilation_w)]

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        dtype (torch.dtype, optional): Data type for the weights and input. Defaults to torch.float16.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        ).to(dtype)

    def forward(self, x, y) -> torch.Tensor:
        z= self.conv1d(x)
        a= self.conv1d(y)
        return z + a

def get_model(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=False, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype)

def get_default_input_shapes():
    batch_size = 16
    in_channels = 3
    length = 512
    return [batch_size, in_channels, length]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]

def get_inputs(batch_size=16, in_channels=3, length=512, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, length, dtype=dtype, device='meta')
    y = torch.empty(batch_size, in_channels, length, dtype=dtype, device='meta')
    return [x, y]

def get_real_inputs(batch_size=16, in_channels=3, length=512, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, length, dtype=dtype), torch.randn(batch_size, in_channels, length, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 2
    dilation = 1
    batch_size = 16
    length = 512
    range_in_channels = [8, 1024]
    range_out_channels = [8, 1024]
    range_kernel_size = [1, 8]
    range_stride = [1, 8]
    range_padding = [1, 16]
    range_dilation = [1, 8]
    range_batch_size = [1, 1024]
    range_length = [8, 20480]
    dtype=[int, int, int, int, int, int, int, int]  
    return [in_channels,  out_channels, kernel_size, stride, padding, dilation, batch_size, length],\
            [range_in_channels, range_out_channels, range_kernel_size,
             range_stride, range_padding, range_dilation,
            range_batch_size, range_length], dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, stride, padding, dilation, batch_size, length):
    return [batch_size, in_channels, length], \
           [in_channels, out_channels, kernel_size, stride, padding, dilation]
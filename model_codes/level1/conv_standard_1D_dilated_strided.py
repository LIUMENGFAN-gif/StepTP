import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        dtype (torch.dtype, optional): Data type for the weights and input. Defaults to torch.float16.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, bias=bias
        )
        self.conv1d = self.conv1d.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d(x)

def get_model(in_channels=3, out_channels=64, kernel_size=3, stride=3, dilation=4, bias=False, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, stride, dilation, bias, dtype)

def get_default_input_shapes():
    batch_size = 16
    in_channels = 3
    length = 256
    return [batch_size, in_channels, length]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 3
    dilation = 4
    return [in_channels, out_channels, kernel_size, stride, dilation]

def get_inputs(batch_size=16, in_channels=3, length=256, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, length, dtype=dtype, device='meta')
    return [x]

def get_real_inputs(batch_size=16, in_channels=3, length=256, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, length, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 3
    dilation = 4
    batch_size = 16
    length = 256
    range_in_channels = [1, 512]
    range_out_channels = [1, 512]
    range_kernel_size = [1, 8]
    range_stride = [1, 8]
    range_dilation = [1, 8]
    range_batch_size = [8, 1024]
    range_length = [1, 2048]
    dtype=[int, int, int, int, int, int, int]
    return [in_channels, out_channels, kernel_size, stride, dilation, batch_size, length], \
           [range_in_channels, range_out_channels, range_kernel_size, range_stride, range_dilation, range_batch_size, range_length], \
           dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, stride, dilation, batch_size, length):
    return [batch_size, in_channels, length], \
           [in_channels, out_channels, kernel_size, stride, dilation]
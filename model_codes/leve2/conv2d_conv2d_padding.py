import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False, dtype=torch.float16):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
    
    def forward(self, x, y) -> torch.Tensor:
        z= self.conv2d(x)
        a= self.conv2d(y)
        return z+a

def get_inputs(batch_size, in_channels, height, width, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, height, width, dtype=dtype, device='meta')
    y = torch.empty(batch_size, in_channels, height, width, dtype=dtype, device='meta')
    return [x, y]

def get_default_input_shapes():
    batch_size = 16
    in_channels = 3
    height = 256
    width = 128
    return [batch_size, in_channels, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5)
    return [in_channels, out_channels, kernel_size]

def get_model(in_channels, out_channels, kernel_size, stride=(2, 2), padding=(3,3), dilation=(4, 4), groups=1, bias=False, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype)

def get_real_inputs(batch_size=16, in_channels=3, height=256, width=128, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype), torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 64
    kernel_size_w = 3
    kernel_size_h = 5
    stride = 1
    padding = 2
    dilation = 1
    batch_size = 16
    height = 256
    width = 128
    range_in_channels = [1, 1024]
    range_out_channels = [1, 1024]
    range_kernel_size_w = [1, 8]
    range_kernel_size_h = [1, 8]
    range_stride = [1, 4]
    range_padding = [1, 8]
    range_dilation = [1, 2]
    range_batch_size = [1, 1024]
    range_height = [8, 4089]
    range_width = [8, 4089]
    dtype=[int,int,int,int,int,int,int,int,int, int]
    return [in_channels, out_channels, kernel_size_w, kernel_size_h, stride, padding, dilation, batch_size, height, width], \
           [range_in_channels, range_out_channels, range_kernel_size_w, range_kernel_size_h, 
            range_stride, range_padding,
            range_dilation, range_batch_size, range_height, range_width], \
           dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size_w, kernel_size_h, stride, padding, dilation, batch_size, height, width):
    return [batch_size, in_channels,  height, width], \
           [in_channels, out_channels, (kernel_size_h, kernel_size_w), (stride, stride), (padding, padding), (dilation,dilation)]
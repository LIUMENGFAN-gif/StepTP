import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
        dtype (torch.dtype, optional): Data type for the weights and input. Defaults to torch.float16.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False, dtype: torch.dtype = torch.float16):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2d = self.conv2d.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)

def get_model(in_channels=3, out_channels=64, bias=False, dtype=torch.float16):
    return Model(in_channels, out_channels, bias, dtype)

def get_default_input_shapes():
    batch_size = 16
    in_channels = 3
    height = 256
    width = 256
    return [batch_size, in_channels, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 64
    return [in_channels, out_channels]

def get_inputs(batch_size=16, in_channels=3, height=256, width=256, dtype=torch.float16):
    x = torch.empty(batch_size, in_channels, height, width, dtype=dtype, device='meta')
    return [x]

def get_real_inputs(batch_size=16, in_channels=3, height=256, width=256, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 64
    batch_size = 16
    height = 256
    width = 256
    range_in_channels = [1, 512]
    range_out_channels = [1, 512]
    range_batch_size = [1, 1024]
    range_height = [8, 2048]
    range_width =[8, 2048]
    dtype=[int, int, int, int, int]
    return [in_channels, out_channels, batch_size, height, width], \
           [range_in_channels, range_out_channels, range_batch_size, range_height, range_width], \
           dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, batch_size, height, width):
    return [batch_size, in_channels, height, width], \
           [in_channels, out_channels]
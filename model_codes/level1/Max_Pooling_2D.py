import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool(x)

def get_inputs(batch_size, channels, height, width, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, channels, height, width, dtype=dtype)
        return [x]
    else:
        return None

def get_default_input_shapes():
    batch_size = 16
    channels = 32
    height = 128
    width = 128
    return [batch_size, channels, height, width]

def get_default_model_params_shapes():
    kernel_size = 2
    stride = 2
    padding = 1 # Padding for height and width
    dilation = 3
    return [kernel_size, stride, padding, dilation]

def get_model(kernel_size, stride, padding, dilation, dtype=torch.float16):
    return Model(kernel_size, stride, padding, dilation)

def get_real_inputs(batch_size=16, channels=32, height=128, width=128, dtype=torch.float16):
    return [torch.randn(batch_size, channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    batch_size = 16
    channels = 32
    height = 128
    width = 128
    range_kernel_size = [1, 7]
    range_stride = [1, 4]
    range_padding = [0, 3]
    range_dilation = [1, 3]
    range_batch_size = [1, 1024]
    range_channels = [1, 1024]
    range_height = [1, 1024]
    range_width = [1, 1024]
    dtype = [int]*8
    return [batch_size, channels, height, width, kernel_size, stride, padding, dilation], \
           [range_batch_size, range_channels, range_height, range_width, range_kernel_size, range_stride, range_padding, range_dilation], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, channels, height, width, kernel_size, stride, padding, dilation):
    return [batch_size, channels, height, width], [kernel_size, stride, padding, dilation]
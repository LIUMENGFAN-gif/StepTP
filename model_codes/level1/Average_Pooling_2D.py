import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool(x)

def get_inputs(batch_size, channels, height, width, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*channels*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, channels, height, width, dtype=dtype)
        return [x]
    else:
        return None

def get_default_input_shapes():
    batch_size = 16
    channels = 64
    height = 256
    width = 256
    return [batch_size, channels, height, width]

def get_default_model_params_shapes():
    kernel_size = 3
    return [kernel_size]

def get_model(kernel_size, dtype=torch.float16):
    return Model(kernel_size)

def get_real_inputs(batch_size=16, channels=64, height=256, width=256, dtype=torch.float16):
    return [torch.randn(batch_size, channels, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    kernel_size = 3
    batch_size = 16
    channels = 64
    height = 256
    width = 256
    range_kernel_size = [2, 8]
    range_batch_size = [1, 1024]
    range_channels = [1, 2048]
    range_height = [1, 2048]
    range_width = [1, 2048]
    dtype = [int, int, int, int,int]
    return [kernel_size, batch_size, channels, height, width], \
           [range_kernel_size, range_batch_size, range_channels, range_height, range_width], dtype

def split_shapes_into_input_and_model_params_shapes(kernel_size, batch_size, channels, height, width):
    return [batch_size, channels, height, width], \
           [kernel_size]
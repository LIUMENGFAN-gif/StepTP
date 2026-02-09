import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return self.avg_pool(x)

def get_inputs(batch_size, in_channels, input_length, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*input_length)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, in_channels, input_length, dtype=dtype)
        return [x]
    else:
        return None

def get_default_input_shapes():
    batch_size = 16
    in_channels = 32
    input_length = 128
    return [batch_size, in_channels, input_length]

def get_default_model_params_shapes():
    kernel_size = 4
    stride = 2
    padding = 1
    return [kernel_size, stride, padding]

def get_model(kernel_size, stride=2, padding=1, dtype=torch.float16):
    return Model(kernel_size, stride, padding)

def get_real_inputs(batch_size=16, in_channels=32, input_length=128, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, input_length, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    kernel_size = 4
    stride = 2
    padding = 1
    batch_size = 16
    in_channels = 32
    input_length = 128
    range_kernel_size = [2, 8]
    range_stride = [1, 4]
    range_padding = [0, 2]
    range_batch_size = [8, 1024]
    range_in_channels = [8, 2048]
    range_input_length = [64, 4096]
    dtype = [int, int, int, int, int, int]
    return [kernel_size, stride, padding, batch_size, in_channels, input_length], \
           [range_kernel_size, range_stride, range_padding, range_batch_size, range_in_channels, range_input_length], dtype

def split_shapes_into_input_and_model_params_shapes(kernel_size, stride, padding, batch_size, in_channels, input_length):
    return [batch_size, in_channels, input_length], \
           [kernel_size, stride, padding]
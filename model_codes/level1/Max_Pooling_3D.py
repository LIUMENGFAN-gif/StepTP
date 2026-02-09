import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 3D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return self.maxpool(x)

def get_inputs(batch_size, channels, dim1, dim2, dim3, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*channels*dim1*dim2*dim3)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, channels, dim1, dim2, dim3, dtype=dtype)
        return [x]
    else:
        return None

def get_default_input_shapes():
    batch_size = 16
    channels = 32
    dim1 = 64
    dim2 = 64
    dim3 = 64
    return [batch_size, channels, dim1, dim2, dim3]

def get_default_model_params_shapes():
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 3
    return_indices = False
    ceil_mode = False
    return [kernel_size, stride, padding, dilation, return_indices, ceil_mode]

def get_model(kernel_size, stride=2, padding=1, dilation=3, return_indices=False, ceil_mode=False, dtype=torch.float16):
    return Model(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

def get_real_inputs(batch_size=16, channels=32, dim1=64, dim2=64, dim3=64, dtype=torch.float16):
    return [torch.randn(batch_size, channels, dim1, dim2, dim3, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 3
    batch_size = 16
    channels = 32
    dim1 = 64
    dim2 = 64
    dim3 = 64
    range_kernel_size = [1, 7]
    range_stride = [1, 4]
    range_padding = [0, 3]
    range_dilation = [1, 4]
    range_batch_size = [1, 1024]
    range_channels = [1, 2048]
    range_dim1 = [1, 2048]
    range_dim2 = [1, 2048]
    range_dim3 = [1, 2048]
    dtype = [int] * 9
    return [batch_size, channels, dim1, dim2, dim3, kernel_size, stride, padding, dilation], \
           [range_batch_size, range_channels, range_dim1, range_dim2, range_dim3, range_kernel_size, range_stride, range_padding, range_dilation], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, channels, dim1, dim2, dim3, kernel_size, stride, padding, dilation):
    return [batch_size, channels, dim1, dim2, dim3], [kernel_size, stride, padding, dilation]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        return self.maxpool(x)

def get_inputs(batch_size, features, sequence_length, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*features*sequence_length)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, features, sequence_length, dtype=dtype)
        return [x]
    else:
        return None

def get_default_input_shapes():
    batch_size = 16
    features = 64
    sequence_length = 128
    return [batch_size, features, sequence_length]

def get_default_model_params_shapes():
    kernel_size = 4
    stride = 2
    padding = 2
    dilation = 3
    return_indices = False
    return [kernel_size, stride, padding, dilation, return_indices]

def get_model(kernel_size, stride=2, padding=2, dilation=3, return_indices=False, dtype=torch.float16):
    return Model(kernel_size, stride, padding, dilation, return_indices)

def get_real_inputs(batch_size=16, features=64, sequence_length=128, dtype=torch.float16):
    return [torch.randn(batch_size, features, sequence_length, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 16
    features = 64
    sequence_length = 128
    kernel_size = 4
    stride = 2
    padding = 2
    dilation = 3
    range_batch_size = [1, 512]
    range_features = [1, 512]
    range_sequence_length = [8, 512]
    range_kernel_size = [1, 8]
    range_stride = [1, 8]
    range_padding = [0, 8]
    range_dilation = [1, 4]
    dtype = [int]*7
    return [batch_size, features, sequence_length, kernel_size,  stride, padding, dilation], \
           [range_batch_size, range_features, range_sequence_length, range_kernel_size, range_stride, range_padding, range_dilation], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, features, sequence_length, kernel_size, stride, padding, dilation):
    return [batch_size, features, sequence_length], [kernel_size, stride, padding, dilation]
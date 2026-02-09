import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return self.avg_pool(x)
    
# PyTorch 的 AvgPool3d 不支持 float16 (torch.float16, 'Half') 输入，尤其是在 CPU 上。
def get_inputs(batch_size, channels, depth, height, width, dtype=torch.float32):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*channels*height*width*depth)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, channels, depth, height, width, dtype=dtype)
        return [x]
    else:
        return None

def get_default_input_shapes():
    batch_size = 16
    channels = 32
    depth = 64
    height = 64
    width = 64
    return [batch_size, channels, depth, height, width]

def get_default_model_params_shapes():
    kernel_size = 3
    stride = 2
    padding = 1
    return [kernel_size, stride, padding]

def get_model(kernel_size, stride=2, padding=1, dtype=torch.float16):
    return Model(kernel_size, stride, padding)

def get_real_inputs(batch_size=16, channels=32, depth=64, height=64, width=64, dtype=torch.float32):
    return [torch.randn(batch_size, channels, depth, height, width, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = 16
    channels = 32
    depth = 64
    height = 64
    width = 64
    range_kernel_size = [2, 8]
    range_stride = [1, 4]
    range_padding = [0, 3]
    range_batch_size = [1, 1024]
    range_channels = [1, 2048]
    range_depth = [1, 2048]
    range_height = [1, 2048]
    range_width = [1, 2048]
    dtype = [int, int, int, int, int, int, int, int]
    
    return [kernel_size, stride, padding, batch_size, channels, depth, height, width],\
            [range_kernel_size, range_stride, range_padding,
             range_batch_size, range_channels,
             range_depth, range_height, range_width],\
            dtype

def split_shapes_into_input_and_model_params_shapes(kernel_size, stride, padding, batch_size, channels, depth, height, width):
    return [batch_size, channels, depth, height, width], \
           [kernel_size, stride, padding]  # Model parameters are kernel_size, stride and padding

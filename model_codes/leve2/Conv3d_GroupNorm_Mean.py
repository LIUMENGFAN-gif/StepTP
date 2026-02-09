import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean.
    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        num_groups (int): Number of groups for GroupNorm.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, dtype=torch.float16):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.group_norm = nn.GroupNorm(num_groups, out_channels).to(dtype)


    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = x.mean(dim=[1, 2, 3, 4]) # Compute mean across all dimensions except batch
        return x

def get_default_input_shapes():
    batch_size = 128
    in_channels = 3
    D = 16
    H = 32
    W = 32
    return [batch_size, in_channels, D, H, W]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    num_groups = 8
    return [in_channels, out_channels, kernel_size, num_groups]

def get_inputs(batch_size=128, in_channels=3, D=16, H=32, W=32, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*D*H*W)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, D, H, W, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=3, num_groups=8, dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, num_groups, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, D=16, H=32, W=32, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, D, H, W, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    num_groups = 8
    batch_size = 128
    D = 16
    H = 32
    W = 32
    range_in_channels = [1, 64]
    range_out_channels = [8,16,64]
    range_kernel_size = [1, 5]
    range_num_groups = [2,4,8]
    range_batch_size = [1, 128]
    range_D = [1, 64]
    range_H = [1, 64]
    range_W = [1, 64]
    dtypes = [int, int, int, int, int, int, int,int]
    return [in_channels, out_channels, kernel_size, num_groups,
            batch_size, D, H, W],\
           [range_in_channels, range_out_channels, range_kernel_size, range_num_groups,
            range_batch_size, range_D, range_H, range_W], dtypes

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, num_groups, batch_size, D, H, W):
    return [batch_size, in_channels, D, H, W], \
           [in_channels, out_channels, kernel_size, num_groups]
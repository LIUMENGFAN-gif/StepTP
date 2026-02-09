import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape,dtype=torch.float16):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False).to(dtype)
        self.multiplier = nn.Parameter(torch.randn(*multiplier_shape, dtype=dtype))
        self.instance_norm = nn.InstanceNorm3d(out_channels, affine=True).to(dtype)

    def forward(self, x, clamp_min, clamp_max):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = torch.clamp(x, clamp_min, clamp_max)
        x = x * self.multiplier
        x = torch.max(x, dim=1)[0]
        return x

def get_default_input_shapes():
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 32, 32
    return [batch_size, in_channels, depth, height, width]

def get_default_model_params_shapes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    multiplier_shape = [out_channels, 1, 1, 1]
    clamp_min = -1.0
    clamp_max = 1.0
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]

def get_inputs(batch_size=128, in_channels=3, depth=16, height=32, width=32,clamp_min=-1.0, clamp_max=1.0,  dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_channels*depth*height*width)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_channels, depth, height, width, dtype=dtype), torch.tensor(clamp_min, dtype=dtype),torch.tensor(clamp_max, dtype=dtype)]
    else:
        return None

def get_model(in_channels=3, out_channels=16, kernel_size=3, multiplier_shape=[16, 1, 1, 1],  dtype=torch.float16):
    return Model(in_channels, out_channels, kernel_size, multiplier_shape, dtype=dtype)

def get_real_inputs(batch_size=128, in_channels=3, depth=16, height=32, width=32,clamp_min=-1.0, clamp_max=1.0, dtype=torch.float16):
    return [torch.randn(batch_size, in_channels, depth, height, width, dtype=dtype), torch.tensor(clamp_min, dtype=dtype),torch.tensor(clamp_max, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    clamp_min = -1.0
    clamp_max = 1.0
    batch_size = 128
    depth = 16
    height = 32
    width = 32
    range_in_channels = [1, 64]
    range_out_channels = [1, 64]
    range_kernel_size = [1, 5]
    range_clamp_min = [-10.0, -0.1]
    range_clamp_max = [0.1, 10.0]
    range_batch_size = [1, 128]
    range_depth = [1, 64]
    range_height = [1, 64]
    range_width = [1, 64]
    dtype=[int, int, int, float, float, int, int, int, int]
    
    return [in_channels, out_channels, kernel_size, clamp_min, clamp_max,
             batch_size, depth, height, width],\
            [range_in_channels, range_out_channels, range_kernel_size,
             range_clamp_min, range_clamp_max,
             range_batch_size, range_depth, range_height, range_width], dtype

def split_shapes_into_input_and_model_params_shapes(in_channels, out_channels, kernel_size, clamp_min, clamp_max, batch_size, depth, height, width):
    return [batch_size, in_channels, depth, height, width, clamp_min, clamp_max], \
           [in_channels, out_channels, kernel_size, [out_channels, 1, 1, 1]]
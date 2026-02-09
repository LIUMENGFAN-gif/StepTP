import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int, dtype=torch.float16):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return self.gn(x)


def get_inputs(batch_size, features, dim1, dim2, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*features*dim1*dim2)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, features, dim1, dim2, dtype=dtype)
        return [x]
    else:
        return None

def get_default_input_shapes():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 128
    return [batch_size, features, dim1, dim2]

def get_default_model_params_shapes():
    num_groups = 8
    features = 64
    return [features, num_groups]

def get_model(features, num_groups, dtype=torch.float16):
    return Model(features, num_groups, dtype)

def get_real_inputs(batch_size=16, features=64, dim1=256, dim2=256, dtype=torch.float16):
    x = torch.randn(batch_size, features, dim1, dim2, dtype=dtype)
    return [x]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 128
    num_groups =8
    range_batch_size = [8,16,32,64,128,256,512,1024]
    range_features = [2,8,16,32,64,128,256,512,1024]
    range_dim1 = [8, 4096]
    range_dim2 = [8, 4096]
    range_num_groups = [2, 4, 8]
    dtype = [int]*5
    return [batch_size, features, dim1, dim2, num_groups], \
           [range_batch_size, range_features, range_dim1, range_dim2, range_num_groups], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, features, dim1, dim2, num_groups):
    return [batch_size, features, dim1, dim2], [features, num_groups]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int, dtype):
        """
        Initializes the BatchNorm layer.
        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        output=self.bn(x)
        return output

def get_default_input_shapes():
    """
    Returns the default input shapes for the model.
    The shape is (batch_size, features, dim1, dim2).
    """
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 128
    return [batch_size, features, dim1, dim2]

def get_default_model_params_shapes():
    features = 64
    return [features]

def get_inputs(batch_size, features, dim1, dim2, dtype=torch.float16):
    """
    Generates input tensor for testing.
    Returns:
        list: [x] where x is (batch_size, features, dim1, dim2)
    """
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*features*dim1*dim2)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, features, dim1, dim2, dtype=dtype)
        return [x]
    else:
        return None

def get_model(features, dtype=torch.float16):
    return Model(features, dtype=dtype)

def get_real_inputs(batch_size=16, features=64, dim1=256, dim2=256, dtype=torch.float16):
    return [torch.randn(batch_size, features, dim1, dim2, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    features = 64
    batch_size = 16
    dim1 = 256
    dim2 = 128
    range_features = [1, 1024]
    range_batch_size = [1, 2048]
    range_dim1 = [1, 2048]
    range_dim2 = [1, 1024]
    dtype = [int, int, int, int]
    return [features, batch_size, dim1, dim2], \
           [range_features, range_batch_size, range_dim1, range_dim2], \
           dtype

def split_shapes_into_input_and_model_params_shapes(features, batch_size, dim1, dim2):
    return [batch_size, features, dim1, dim2], \
           [features]
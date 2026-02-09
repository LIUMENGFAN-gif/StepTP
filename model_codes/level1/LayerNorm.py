import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple, dtype=torch.float16):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)

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
    features = 64
    dim1 = 256
    dim2 = 128
    return [(features, dim1, dim2)]

def get_model(normalized_shape, dtype=torch.float16):
    return Model(normalized_shape, dtype)

def get_real_inputs(batch_size=16, features=64, dim1=256, dim2=256, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*features*dim1*dim2)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.empty(batch_size, features, dim1, dim2, dtype=dtype, device='meta')]
    else:
        return None

def set_default_shapes_ranges_and_dtypes():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 128
    range_batch_size = [1, 1024]
    range_features = [1, 1024]
    range_dim1 = [1, 1024]
    range_dim2 = [1, 1024]
    dtype = [int]*4
    return [batch_size, features, dim1, dim2], [range_batch_size, range_features, range_dim1, range_dim2], dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, features, dim1, dim2):
    return [batch_size, features, dim1, dim2], [(features, dim1, dim2)]
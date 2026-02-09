import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        norm = torch.norm(x, p='fro')
        return x / norm



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
    return []

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(batch_size=16, features=64, dim1=256, dim2=256, dtype=torch.float16):
    x = torch.randn(batch_size, features, dim1, dim2, dtype=dtype)
    return [x]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 128
    range_batch_size = [1, 1024]
    range_features = [1, 4096]
    range_dim1 = [1, 4096]
    range_dim2 = [1, 4096]
    dtype = [int]*4
    return [batch_size, features, dim1, dim2], [range_batch_size, range_features, range_dim1, range_dim2], dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, features, dim1, dim2):
    return [batch_size, features, dim1, dim2], []
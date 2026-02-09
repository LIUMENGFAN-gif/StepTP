import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, dim, *).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return x / torch.norm(x, p=2, dim=1, keepdim=True)


def get_inputs(batch_size, dim, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*dim)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, dim, dtype=dtype)
        return [x]
    else:
        return None

def get_default_input_shapes():
    batch_size = 16
    dim = 16384
    return [batch_size, dim]

def get_default_model_params_shapes():
    return []

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(batch_size=16, dim=16384, dtype=torch.float16):
    x = torch.randn(batch_size, dim, dtype=dtype)
    return [x]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 16
    dim = 16384
    range_batch_size = [1, 1024]
    range_dim = [1, 16384*2]
    dtype = [int, int]
    
    return [batch_size, dim], \
           [range_batch_size, range_dim], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, dim):
    return [batch_size, dim], []
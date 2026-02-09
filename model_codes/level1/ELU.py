import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.
        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of any shape.
        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        return F.elu(x, alpha=self.alpha)

def get_default_input_shapes():
    """
    Returns the default input shapes for the model.
    The shape is (batch_size, dim).
    """
    batch_size = 16
    dim = 16384
    return [batch_size, dim]

def get_default_model_params_shapes():
    return []

def get_inputs(batch_size, dim, dtype=torch.float16):
    """
    Generates input tensor for testing.
    Returns:
        list: [x] where x is (batch_size, dim) on device='meta'.
    """
    x = torch.empty(batch_size, dim, dtype=dtype, device='meta')
    return [x]

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
    return [batch_size, dim], [range_batch_size, range_dim], dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, dim):
    return [batch_size, dim], []
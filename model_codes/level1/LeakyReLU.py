import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.
        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(Model, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of any shape.
        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)

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
    
    return [batch_size, dim], \
           [range_batch_size, range_dim], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, dim):
    return [batch_size, dim], []
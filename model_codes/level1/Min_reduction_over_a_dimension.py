import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return torch.min(x, dim=self.dim)[0]

def get_inputs(batch_size, dim1, dim2, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*dim1*dim2)/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, dim1, dim2, dtype=dtype)
        return [x]
    else:
        return None

def get_default_input_shapes():
    batch_size = 16
    dim1 = 256
    dim2 = 256
    return [batch_size, dim1, dim2]

def get_default_model_params_shapes():
    reduce_dim = 1
    return [reduce_dim]

def get_model(reduce_dim=1, dtype=torch.float16):
    return Model(dim=reduce_dim)

def get_real_inputs(batch_size=16, dim1=256, dim2=256, dtype=torch.float16):
    x = torch.randn(batch_size, dim1, dim2, dtype=dtype)
    return [x]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 16
    dim1 = 256
    dim2 = 256
    reduce_dim = 1
    range_batch_size = [1, 1024]
    range_dim1 = [1, 1024]
    range_dim2 = [1, 1024]
    range_reduce_dim = [0, 2]  # Assuming reduction can be over dim1 or dim2
    dtype = [int, int, int, int]
    return [batch_size, dim1, dim2, reduce_dim], \
           [range_batch_size, range_dim1, range_dim2, range_reduce_dim], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, dim1, dim2, reduce_dim):
    return [batch_size, dim1, dim2], [reduce_dim]
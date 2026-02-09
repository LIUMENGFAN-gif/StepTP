import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return torch.argmax(x, dim=self.dim)

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
    return [torch.randn(batch_size, dim1, dim2, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    reduce_dim = 1
    batch_size = 16
    dim1 = 256
    dim2 = 256
    range_reduce_dim = [0, 2]  # Assuming the input tensor has at least 3 dimensions
    range_batch_size = [1, 1024]
    range_dim1 = [1, 1024]
    range_dim2 = [1, 1024]
    dtype = [int, int, int, int]
    return [reduce_dim, batch_size, dim1, dim2], \
           [range_reduce_dim, range_batch_size, range_dim1, range_dim2], dtype

def split_shapes_into_input_and_model_params_shapes(reduce_dim, batch_size, dim1, dim2):
    return [batch_size, dim1, dim2], \
           [reduce_dim]
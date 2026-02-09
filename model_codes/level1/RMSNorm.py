import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, dtype=torch.float16):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = torch.tensor(eps,dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the feature dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

        # Normalize the input by dividing by the RMS
        return x / rms



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
    return [features]

def get_model(num_features, eps=1e-5, dtype=torch.float16):
    return Model(num_features, eps, dtype)

def get_real_inputs(batch_size=16, features=64, dim1=256, dim2=256, dtype=torch.float16):
    x = torch.randn(batch_size, features, dim1, dim2, dtype=dtype)
    return [x]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 128
    eps = 1e-5
    range_batch_size = [1, 1024]
    range_features = [1, 4096]
    range_dim1 = [1, 4096]
    range_dim2 = [1, 4096]
    range_eps = [1e-6, 1e-3]  # Example range for epsilon
    dtype = [int, int, int, int, float]
    return [batch_size, features, dim1, dim2, eps], \
           [range_batch_size, range_features, range_dim1, range_dim2, range_eps], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, features, dim1, dim2, eps):
    return [batch_size, features, dim1, dim2], [features, eps]
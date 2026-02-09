import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    Parameters:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        eps (float): Epsilon for instance normalization.
        momentum (float): Momentum for instance normalization.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1, dtype=torch.float16):
        super().__init__()
        self.bmm = nn.Linear(in_features, out_features).to(dtype)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum, affine=True).to(dtype)
        self.dtype = dtype

    def forward(self, x, y):
        x = self.bmm(x)
        x=x.unsqueeze(-1).unsqueeze(-1)
        x = torch.cat((x,x),dim=-1)#.unsqueeze(1) . # (batch_size, out_features, 1, 1)
        x = self.instance_norm(x)
        x = x.squeeze(-2)#.squeeze(1)
        x = x + y
        x = x * y
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 64
    out_features = 256
    return [batch_size, in_features, out_features]

def get_default_model_params_shapes():
    in_features = 64
    out_features = 256
    eps = 1e-5
    momentum = 0.1
    return [in_features, out_features, eps, momentum]

def get_inputs(batch_size=128, in_features=64, out_features=128, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*(in_features+out_features*2))/ (1024 ** 3)
    if tensor_memory_gb<30:
        x = torch.randn(batch_size, in_features, dtype=dtype)
        y = torch.randn((batch_size, out_features, 2), dtype=dtype)
        return [x, y]
    else:
        return None

def get_model(in_features=64, out_features=128, eps=1e-5, momentum=0.1, dtype=torch.float16):
    return Model(in_features, out_features, eps=eps, momentum=momentum, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=64, out_features=128, dtype=torch.float16):
    x = torch.randn(batch_size, in_features, dtype=dtype)
    y = torch.randn((batch_size, out_features, 2), dtype=dtype)
    return [x, y]

def set_default_shapes_ranges_and_dtypes():
    in_features = 64
    out_features = 256
    eps = 1e-5
    momentum = 0.1
    batch_size = 128
    range_in_features = [8, 2048]
    range_out_features = [8, 2048]
    range_eps = [1e-6, 1e-4]
    range_momentum = [0.01, 0.99]
    range_batch_size = [1, 1024]
    dtypes=[int, int, float, float, int]
    return [in_features, out_features, eps, momentum, batch_size], [range_in_features, range_out_features, range_eps, range_momentum, range_batch_size], dtypes

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, eps, momentum, batch_size):
    return [batch_size, in_features, out_features], [in_features, out_features, eps, momentum]




# if __name__ == "__main__":
#     # Example usage
#     input_shape= get_default_input_shapes()
#     model_shape= get_default_model_params_shapes()
#     model = get_model(*model_shape)
#     inputs = get_inputs(*input_shape)
#     output = model(*inputs)
#     # print(f"Output shape: {output.shape}")
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01, dtype=torch.float16):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size).to(dtype)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps).to(dtype)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope).to(dtype)

    def forward(self, x):
        x = self.fc(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        x = x + x
        return x

def get_default_input_shapes():
    batch_size = 128
    input_size = 512
    return [batch_size, input_size]

def get_default_model_params_shapes():
    input_size = 512
    hidden_size = 256
    num_groups = 8
    return [input_size, hidden_size, num_groups]

def get_inputs(batch_size, input_size, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_size) / (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, input_size, dtype=dtype)]
    else:
        return None

def get_model(input_size, hidden_size, num_groups, dtype=torch.float16):
    return Model(input_size, hidden_size, num_groups, dtype=dtype)

def get_real_inputs(batch_size=128, input_size=512, dtype=torch.float16):
    return [torch.randn(batch_size, input_size, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    input_size = 512
    hidden_size = 256
    num_groups = 8
    batch_size = 128
    range_input_size = [64, 4096]
    range_hidden_size = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    range_num_groups = [2,4,8]
    range_batch_size = [1, 1024]
    dtype = [int, int, int, int]
    return [input_size, hidden_size, num_groups, batch_size], \
           [range_input_size, range_hidden_size, range_num_groups, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(input_size, hidden_size, num_groups, batch_size):
    return [batch_size, input_size], \
           [input_size, hidden_size, num_groups]
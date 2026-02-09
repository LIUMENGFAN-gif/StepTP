import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, scaling_factor, dtype=torch.float16):
        super(Model, self).__init__()
        self.gemm = nn.Linear(input_size, input_size).to(dtype)
        self.scaling_factor = torch.tensor(scaling_factor, dtype=dtype)

    def forward(self, x):
        original_x = x
        x = self.gemm(x)
        x = torch.sigmoid(x)
        x = x * self.scaling_factor
        x = x + original_x
        return x

def get_default_input_shapes():
    batch_size = 128
    input_size = 1024
    return [batch_size, input_size]

def get_default_model_params_shapes():
    input_size = 1024
    scaling_factor = 2.0
    return [input_size, scaling_factor]

def get_inputs(batch_size, input_size, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_size)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, input_size, dtype=dtype)]
    else:
        return None

def get_model(input_size, scaling_factor, dtype=torch.float16):
    return Model(input_size, scaling_factor, dtype=dtype)

def get_real_inputs(batch_size=128, input_size=1024, dtype=torch.float16):
    return [torch.randn(batch_size, input_size, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    input_size = 1024
    scaling_factor = 2.0
    batch_size = 128
    range_input_size = [8, 2048]
    range_scaling_factor = [0.1, 10.0]
    range_batch_size = [1, 1024]
    dtype = [int, float, int]
    return [input_size, scaling_factor, batch_size], \
           [range_input_size, range_scaling_factor, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(input_size, scaling_factor, batch_size):
    return [batch_size, input_size], \
           [input_size, scaling_factor]
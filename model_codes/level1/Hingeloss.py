import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
    None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

def get_default_input_shapes():
    batch_size = 128
    input_shape = 1
    return [batch_size, input_shape]

def get_default_model_params_shapes():
    return []

def get_inputs(batch_size, input_shape, dtype=torch.float16):
    return [torch.empty(batch_size, input_shape,dtype=dtype, device='meta'), torch.randint(0, 2, (batch_size, 1),dtype=dtype, device='meta') * 2 - 1]

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(batch_size=128, input_shape=1, dtype=torch.float16):
    predictions = torch.randn(batch_size, input_shape, dtype=dtype)
    targets = torch.randint(0, 2, (batch_size, 1), dtype=dtype) * 2 - 1
    return [predictions, targets]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 128
    input_shape = 1
    range_batch_size = [1, 1024]
    range_input_shape = [1, 16384*2]
    dtype = [int, int]
    return [batch_size, input_shape], \
           [range_batch_size, range_input_shape], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, input_shape):
    return [batch_size, input_shape], []
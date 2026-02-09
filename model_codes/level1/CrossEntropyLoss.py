import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, dtype=torch.float16):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets)

def get_default_input_shapes():
    batch_size = 1024
    num_classes = 10
    return [batch_size, num_classes]

def get_default_model_params_shapes():
    # No learnable parameters for CrossEntropyLoss
    return []

def get_inputs(batch_size=4096, input_shape=10, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (2*batch_size*input_shape)/ (1024 ** 3)
    if tensor_memory_gb<30:
        predictions = torch.randn(batch_size, input_shape, dtype=dtype)
        targets = torch.randint(0, input_shape, (batch_size,), dtype=torch.long)
        return [predictions, targets]
    else:
        return None

def get_model(shape=None,dtype=torch.float16):
    return Model(dtype=dtype)

def get_real_inputs(batch_size=4096, input_shape=10, dtype=torch.float16):
    predictions = torch.randn(batch_size, input_shape, dtype=dtype)
    targets = torch.randint(0, input_shape, (batch_size,), dtype=torch.long)
    return [predictions, targets]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 1024
    input_shape = 10
    range_batch_size = [1, 1024]
    range_input_shape = [2, 1024]
    dtype = [int, int]
    return [batch_size, input_shape], \
           [range_batch_size, range_input_shape], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, input_shape):
    return [batch_size, input_shape], []
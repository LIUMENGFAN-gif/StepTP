import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Cosine Similarity Loss for comparing vectors.

    Parameters:
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, dtype=torch.float16):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, predictions, targets):
        cosine_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
        return torch.mean(1 - cosine_sim)

def get_default_input_shapes():
    batch_size = 128
    input_shape = 4096
    return [batch_size, input_shape]

def get_default_model_params_shapes():
    # No learnable parameters for CosineSimilarityLoss
    return []

def get_inputs(batch_size=128, input_shape=4096, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (2*batch_size*input_shape)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, input_shape, dtype=dtype), torch.randn(batch_size, input_shape, dtype=dtype)]
    else:
        return None

def get_model(shape=None,dtype=torch.float16):
    return Model(dtype=dtype)

def get_real_inputs(batch_size=128, input_shape=4096, dtype=torch.float16):
    return [torch.randn(batch_size, input_shape, dtype=dtype), torch.randn(batch_size, input_shape, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 128
    input_shape = 4096
    range_batch_size = [1, 1024]
    range_input_shape = [1, 16384]
    dtype = [int, int]
    return [batch_size, input_shape], [range_batch_size, range_input_shape], dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, input_shape):
    return [batch_size, input_shape], []
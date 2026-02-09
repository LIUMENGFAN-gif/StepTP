import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
        dtype (torch.dtype): The data type for computation (default: torch.float16).
    """
    def __init__(self, margin=1.0, dtype=torch.float16):
        super(Model, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)
        self.dtype = dtype

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

def get_default_input_shapes():
    batch_size = 128
    input_shape = 1024
    return [batch_size, input_shape]

def get_default_model_params_shapes():
    # No learnable parameters for TripletMarginLoss
    margin=1.0
    return [margin]

def get_inputs(batch_size=128, input_shape=4096, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*input_shape*3)/ (1024 ** 3)
    if tensor_memory_gb<30:
        anchor = torch.randn(batch_size, input_shape, dtype=dtype)
        positive = torch.randn(batch_size, input_shape, dtype=dtype)
        negative = torch.randn(batch_size, input_shape, dtype=dtype)
        return [anchor, positive, negative]
    else:
        return None

def get_model(margin=1.0, dtype=torch.float16):
    return Model(margin=margin, dtype=dtype)

def get_real_inputs(batch_size=128, input_shape=4096, dtype=torch.float16):
    anchor = torch.randn(batch_size, input_shape, dtype=dtype)
    positive = torch.randn(batch_size, input_shape, dtype=dtype)
    negative = torch.randn(batch_size, input_shape, dtype=dtype)
    return [anchor, positive, negative]

def set_default_shapes_ranges_and_dtypes():
    batch_size = 128
    input_shape = 1024
    margin = 1.0
    range_batch_size = [1, 1024]
    range_input_shape = [1, 4096]
    range_margin = [0.1, 10.0]  # Assuming margin can vary between 0.1 and 10.0
    dtype = [int, int, float]
    return [batch_size, input_shape, margin], \
           [range_batch_size, range_input_shape, range_margin], \
           dtype

def split_shapes_into_input_and_model_params_shapes(batch_size, input_shape, margin):
    return [batch_size, input_shape], [margin]
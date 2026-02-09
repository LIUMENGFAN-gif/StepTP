import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        C=torch.matmul(A, B)
        return torch.softmax(C, dim=1)

def get_default_input_shapes():
    N = 64
    K = 128
    L = 32
    return [N, K, L]

def get_default_model_params_shapes():
    return []


def get_inputs(N, K, L, dtype=torch.float16):
    A = torch.empty(N, K, dtype=dtype, device='meta')
    B = torch.empty(K, L, dtype=dtype, device='meta')
    return [A, B]

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(N=64, K=128, L=32, dtype=torch.float16):
    A = torch.randn(N, K, dtype=dtype)
    B = torch.randn(K, L, dtype=dtype)
    return [A, B]

def set_default_shapes_ranges_and_dtypes():
    N = 64
    K = 128
    L = 32
    range_N = [1, 1024]
    range_K = [1, 16384]
    range_L = [1, 16384]
    dtype = [int, int, int]
    return [N, K, L], \
           [range_N, range_K, range_L], dtype

def split_shapes_into_input_and_model_params_shapes(N, K, L):
    return [N, K, L], \
           []
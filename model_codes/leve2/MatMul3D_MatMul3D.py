import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, C):
        E=torch.matmul(A, B) # shape: (N, M, L)
        F=torch.matmul(C, B) # shape: (N, M, G)
        return E+F



def get_default_input_shapes():
    N = 16
    M = 1024
    K = 2048
    L = 768
    return [N, M, K, L]

def get_default_model_params_shapes():
    return []

def get_inputs(N,M,K,L, dtype=torch.float16):
    A = torch.empty(N, M, K,dtype=dtype, device='meta')
    B = torch.empty(K, L,dtype=dtype, device='meta')
    C = torch.empty(N, M, K,dtype=dtype, device='meta')
    return [A, B, C]

def get_model(shape=None, dtype=torch.float16):
    return Model()

def get_real_inputs(N=16, M=1024, K=2048, L=768, dtype=torch.float16):
    A = torch.randn(N, M, K, dtype=dtype)
    B = torch.randn(K, L, dtype=dtype)
    C = torch.randn(N, M, K, dtype=dtype)
    return [A, B, C]

def set_default_shapes_ranges_and_dtypes():
    N = 16
    M = 1024
    K = 2048
    L = 768
    range_N = [1, 1024]
    range_M = [1, 16384]
    range_K = [1, 16384]
    range_L = [1, 16384]
    dtype = [int, int, int, int]

    return [N, M, K, L], \
           [range_N, range_M, range_K, range_L], \
           dtype

def split_shapes_into_input_and_model_params_shapes(N, M, K, L):
    return [N, M, K, L], []  # No model parameters in this case
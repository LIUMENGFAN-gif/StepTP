import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=[1], dtype=torch.float16):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features).to(dtype)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum).to(dtype)
        self.scale = nn.Parameter(torch.ones(*scale_shape, dtype=dtype))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.gemm(x)
        x = self.bn(x)
        x = self.scale * x
        x = self.softmax(x)
        return x

def get_default_input_shapes():
    batch_size = 128
    in_features = 1024
    return [batch_size, in_features]

def get_default_model_params_shapes():
    in_features = 1024
    out_features = 512
    bn_eps = 1e-5
    bn_momentum = 0.1
    scale_shape = [1]
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]

def get_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    tensor_memory_gb = torch.tensor([1], dtype=dtype).element_size() * (batch_size*in_features)/ (1024 ** 3)
    if tensor_memory_gb<30:
        return [torch.randn(batch_size, in_features, dtype=dtype)]
    else:
        return None

def get_model(in_features=1024, out_features=512, bn_eps=1e-5, bn_momentum=0.1, scale_shape=[1], dtype=torch.float16):
    return Model(in_features, out_features, bn_eps, bn_momentum, scale_shape, dtype=dtype)

def get_real_inputs(batch_size=128, in_features=1024, dtype=torch.float16):
    return [torch.randn(batch_size, in_features, dtype=dtype)]

def set_default_shapes_ranges_and_dtypes():
    in_features = 1024
    out_features = 512
    bn_eps = 1e-5
    bn_momentum = 0.1
    batch_size = 128
    range_in_features = [8, 4096]
    range_out_features = [8, 16, 32, 64, 128, 256, 512, 1024]
    range_bn_eps = [1e-6, 1e-3]
    range_bn_momentum = [0.01, 0.99]
    range_batch_size = [1, 1024]
    dtype = [int, int, float, float, int]
    return [in_features, out_features, bn_eps, bn_momentum, batch_size], \
           [range_in_features, range_out_features, range_bn_eps, range_bn_momentum, range_batch_size], dtype

def split_shapes_into_input_and_model_params_shapes(in_features, out_features, bn_eps, bn_momentum, batch_size):
    return [batch_size, in_features], \
           [in_features, out_features, bn_eps, bn_momentum, [1]]
import torch
import json
import importlib
from ops import *
import os
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import inspect
import argparse
import gzip
import pickle

def select_model(model_name, use_default_shape=True, default_input_shapes=None, default_model_params_shapes=None):
    params_name=[]
    params_shape=[]
    params_dtype=[]
    module = importlib.import_module("model_codes")
    get_default_input_shapes = getattr(module, f"{model_name}_get_default_input_shapes", None)
    get_default_model_params_shapes = getattr(module, f"{model_name}_get_default_model_params_shapes", None)
    get_inputs = getattr(module, f"{model_name}_get_inputs")
    get_model = getattr(module, f"{model_name}_get_model")
    default_input_shapes = get_default_input_shapes()
    default_model_params_shapes = get_default_model_params_shapes()
    inputs = get_inputs(*default_input_shapes)
    model = get_model(*default_model_params_shapes)
    model.eval()
    if not isinstance(inputs, tuple):
        inputs = tuple(inputs)
    for name, params in model.named_parameters():
        params_name.append(name)
        params_shape.append(params.shape)
        params_dtype.append(params.dtype)
    return model, inputs, params_name, params_shape, params_dtype

def select_model_for_store(model_name, dtype):
    params_name=[]
    params_shape=[]
    params_dtype=[]
    params_value_dict={}
    module = importlib.import_module("model_codes")
    set_default_shapes_ranges_and_dtypes = getattr(module, f"{model_name}_set_default_shapes_ranges_and_dtypes")
    split_shapes_into_input_and_model_params_shapes = getattr(module, f"{model_name}_split_shapes_into_input_and_model_params_shapes")
    get_inputs = getattr(module, f"{model_name}_get_inputs")
    get_model = getattr(module, f"{model_name}_get_model")
    default_shapes, _, _ = set_default_shapes_ranges_and_dtypes()
    input_shapes, model_params_shapes = split_shapes_into_input_and_model_params_shapes(*default_shapes)
    inputs = get_inputs(*input_shapes, dtype=dtype)
    if inputs is None:
        raise ValueError(f"Inputs for {model_name} are None. Please choose smaller shapes.")
    model = get_model(*model_params_shapes, dtype=dtype)
    model.eval()
    if not isinstance(inputs, tuple):
        inputs = tuple(inputs)
    for name, params in model.named_parameters():
        params_name.append(name)
        params_value_dict[name]=params.detach().cpu()
        params_shape.append(params.shape)
        params_dtype.append(params.dtype)
    return model, inputs, params_name, params_shape, params_dtype, params_value_dict, input_shapes, default_shapes

def generate_IR_related_info(inputs_info, constant_info, params_info, outputs_info):
    known_names= inputs_info[0] + constant_info[0] + params_info[0] + outputs_info[0]
    known_shapes = inputs_info[1] + constant_info[1] + params_info[1] + outputs_info[1]
    known_dtype = inputs_info[2] + constant_info[2] + params_info[2] + outputs_info[2]
    input_known_names= inputs_info[0]+ constant_info[0] + params_info[0]
    return known_names, known_shapes, known_dtype, input_known_names

def serialize_value_dict(value_dict):
    serialized = {}
    for name, value in value_dict.items():
        if isinstance(value, torch.Tensor):
            serialized[name] = [value.detach().cpu(), str(value.dtype)]
        else:
            tensor_value = torch.as_tensor(value)
            serialized[name] = [tensor_value, str(tensor_value.dtype)]
    return serialized

def save_store_dict(store_dict_dir, model_name, model_index, default_shapes, dtype, inputs_info, outputs_info, constant_info, params_info, input_shapes, params_value_dict, constant_value_dict, paramsname_constant_mapping):
    known_names, known_shapes, known_dtype, input_known_names=generate_IR_related_info(inputs_info, constant_info, params_info, outputs_info)
    os.makedirs(store_dict_dir, exist_ok=True)
    store_dict = {
        'shape': list(default_shapes),
        'known_names': known_names,
        'known_shapes': [list(shape) for shape in known_shapes],
        'known_dtype': [str(dtype_item) for dtype_item in known_dtype],
        'input_known_names': input_known_names,
        'input_shapes': list(input_shapes),
        'dtype': str(dtype),
        'output_dtype': str(outputs_info[2][0]),
        'output_shape': list(outputs_info[1][0]),
        'constant_value_dict': serialize_value_dict(constant_value_dict),
        'params_value_dict': serialize_value_dict(params_value_dict),
        'paramsname_constant_mapping': paramsname_constant_mapping,
        'constant_names': constant_info[0],
        'params_names': params_info[0],
    }
    store_dict_path = os.path.join(store_dict_dir, f'{model_name}_original_{model_index}.pkl.gz')
    with gzip.open(store_dict_path, 'wb') as f:
        pickle.dump(store_dict, f)
    return store_dict_path

def parse_torch_dtype(dtype_name):
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name.split(".", 1)[1]
    return getattr(torch, dtype_name)

def analyze_pytorch_model(model, inputs, original_params_name, original_params_shape, original_params_dtype):
    ops = []
    inputs_name, inputs_shape, inputs_dtype = [],[],[]
    params_name, params_shape, params_dtype=[],[],[]
    constant_name, constant_shape, constant_dtype = [],[],[]
    constant_value_list={}
    outputs_name, outputs_shape, outputs_dtype = [],[],[]
    intermediate_names, intermediate_shapes, intermediate_dtypes = [],[],[]
    name_start_idx=0
    placeholder_num=0
    print("Start tracing graph.")
    try:
        traced = symbolic_trace(model)
    except Exception as e:
        print(f"Error during symbolic tracing: {e}")
        print("Retrying with concrete arguments.")
        concrete_args={}
        for _, (name, inp) in enumerate(zip(inspect.signature(model.forward).parameters.keys(), inputs)):
            concrete_args[name] = inp
            placeholder_num+=1
        traced = symbolic_trace(model, concrete_args=concrete_args)
    ShapeProp(traced).propagate(*inputs)
    print("Traced graph done.")
    for node in traced.graph.nodes:
        # print(f"op: {node.op}, name:{node.name}, target: {node.target}, args: {node.args}, kwargs: {node.kwargs}")
        # print("meta:", node.meta)
        placeholder_num, ops, name_start_idx, inputs_name, inputs_shape, inputs_dtype, outputs_name, outputs_shape, outputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, constant_name, constant_shape, constant_dtype, constant_value_list, original_params_name = handle_each_node(traced, placeholder_num, node, ops, name_start_idx, inputs_name, inputs_shape, inputs_dtype, outputs_name, outputs_shape, outputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, constant_name, constant_shape, constant_dtype,constant_value_list, original_params_name, original_params_shape, original_params_dtype)
    # print("before ops:", ops)
    print("handle each node done.")
    ops, inputs_name, outputs_name, intermediate_names, constant_name, params_name, name_start_idx, paramsname_constant_mapping = rename(ops, inputs_name, outputs_name, intermediate_names, constant_name, params_name, name_start_idx)
    return ops, [inputs_name, inputs_shape, inputs_dtype], [outputs_name, outputs_shape, outputs_dtype], [intermediate_names,intermediate_shapes, intermediate_dtypes], [constant_name, constant_shape, constant_dtype], [params_name, params_shape, params_dtype], name_start_idx, paramsname_constant_mapping, constant_value_list

def convert_to_IR(ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx):
    IR=''
    last_mean_info=[]
    has_min_max=False
    len_ops = len(ops)
    for op in ops:
        IR,name_start_idx, last_mean_info, len_ops, has_min_max=handle_each_op(IR, op, len_ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx,last_mean_info, has_min_max)
    if len_ops==0:
        print("IR done.")
        return IR, name_start_idx
    else:
        print("Error: not all ops are handled, len_ops:", len_ops)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace a PyTorch model, convert it to LEIR, and save the verification store dict.")
    parser.add_argument("--level", type=str, default="level1")
    parser.add_argument("--model_index", type=int, default=27)
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--store_dict_dir", type=str, default="store_dict")
    args = parser.parse_args()
    level=args.level
    model_index=args.model_index
    dtype=parse_torch_dtype(args.dtype)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, level+"_model_name.json")
    with open(json_path, "r") as f:
        model_names = json.load(f)
    constant_static=[]
    # for model_index in range(len(model_names)):
    model_name = model_names[model_index]
    print("model_name:", model_name)
    model, inputs, params_name, params_shape, params_dtype, params_value_dict, input_shapes, default_shapes=select_model_for_store(model_name, dtype)
    print("model done")
    # print("model:", model)
    # print("params_name:", params_name)
    ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx, paramsname_constant_mapping, constant_value_dict = analyze_pytorch_model(model, inputs, params_name, params_shape, params_dtype)
    print("analyze done")
    # print("after ops:", ops)
    # print("inputs_info:", inputs_info)
    # print("outputs_info:", outputs_info)
    # print("intermediate_info:", intermediate_info)
    print("constant_info:", constant_info)
    if len(constant_info[0])>0:
        constant_static.append([model_index,model_name, constant_info])
    # print("params_info:", params_info)
    # print("name_start_idx:", name_start_idx)
    # print("paramsname_constant_mapping:", paramsname_constant_mapping)
    # print("model_name:", model_name)
    IR, name_start_idx = convert_to_IR(ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx)
    store_dict_path = save_store_dict(args.store_dict_dir, model_name, model_index, default_shapes, dtype, inputs_info, outputs_info, constant_info, params_info, input_shapes, params_value_dict, constant_value_dict, paramsname_constant_mapping)
    print(f"store_dict saved to: {store_dict_path}")
    print(f'IR: {IR}')
    print(f"model_name:{model_name}, {model_index}/{len(model_names)}")
    print("constant_static:", constant_static)

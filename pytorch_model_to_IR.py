import torch
import json
import importlib
from ops import *
import os
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import inspect

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
    len_ops = len(ops)
    for op in ops:
        IR,name_start_idx, last_mean_info, len_ops=handle_each_op(IR, op, len_ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx,last_mean_info)
    if len_ops==0:
        print("IR done.")
        return IR, name_start_idx
    else:
        print("Error: not all ops are handled, len_ops:", len_ops)

if __name__ == "__main__":
    level='level1'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, level+"_model_name.json")
    with open(json_path, "r") as f:
        model_names = json.load(f)
    model_index=27
    constant_static=[]
    # for model_index in range(len(model_names)):
    model_name = model_names[model_index]
    print("model_name:", model_name)
    model, inputs, params_name, params_shape, params_dtype=select_model(model_name)
    print("model done")
    # print("model:", model)
    # print("params_name:", params_name)
    ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx, paramsname_constant_mapping, constant_value_list = analyze_pytorch_model(model, inputs, params_name, params_shape, params_dtype)
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
    IR= convert_to_IR(ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx)
    print(f'IR: {IR}')
    print(f"model_name:{model_name}, {model_index}/{len(model_names)}")
    print("constant_static:", constant_static)
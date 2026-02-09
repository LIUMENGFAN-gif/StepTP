from ops import *
from strategies import *
from pytorch_model_to_IR import *

def apply_strategy_to_IR(IR, name_start_idx, strategy_name, num, input_output_name):
    row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops,eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops = split_IR_to_equations(IR)
    # print("loops:", loops)
    # print("equations_under_loops:", equations_under_loops)
    # print("eq_outputs_under_loops:", eq_outputs_under_loops)
    # print("eq_inputs_under_loops:", eq_inputs_under_loops)
    # print("simplified_eqs_under_loops:", simplified_eqs_under_loops)
    # print("simplified_eq_outputs_under_loops:", simplified_eq_outputs_under_loops)
    # print("simplified_eq_inputs_under_loops:", simplified_eq_inputs_under_loops)
    apply_strategy_to_IR = getattr(importlib.import_module("strategies"), f"apply_{strategy_name}_to_IR", None)
    original_IR, transformed_IR_list, has_transformation=apply_strategy_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops,eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    if has_transformation:
        if len(transformed_IR_list)>=len(original_IR):
            num+=len(transformed_IR_list)
        else:
            num+=len(original_IR)
    # print(f'has_transformation:{has_transformation}, transformed_IR_list:{transformed_IR_list}')
    return transformed_IR_list, num

if __name__ == "__main__":
    level='level1'
    strategy_index = 10
    num_list=[]
    # for strategy_index in range(30, 40):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_json_path = os.path.join(script_dir, level+"_model_name.json")
    with open(model_json_path, "r") as f:
        model_names = json.load(f)
    model_index=18
    num=0
    if strategy_index==40 and level=='level2':
        model_name_list=[5,6,7]
    else:
        model_name_list=list(range(len(model_names)))
    model_name_list=[0]
    for model_index in model_name_list:
        print(f"Processing model {model_index}/{len(model_names)}|strategy {strategy_index}")
        model_name = model_names[model_index]
        print("model_name:", model_name)
        model, inputs, params_name, params_shape, params_dtype=select_model(model_name)
        # print("model done")
        # print("model:", model)
        # print("params_name:", params_name, params_shape)
        ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx, paramsname_constant_mapping, constant_value_list = analyze_pytorch_model(model, inputs, params_name, params_shape, params_dtype)
        # print("analyze done")
        # print("model_name:", model_name)
        # print(f'inputs_info:{inputs_info}')
        # print(f'intermediate_info:{intermediate_info}')
        # print(f'params_info:{params_info}')
        input_output_name=inputs_info[0]+outputs_info[0]+constant_info[0]+params_info[0]
        # print("input_output_name:", input_output_name)
        IR, name_start_idx= convert_to_IR(ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx)
        # print("after ops:", ops)
        print(f'orginal IR: {IR}')
        print(f"model_name:{model_name}, {model_index}/{len(model_names)}|strategy {strategy_index}")
        strategy_json_path = os.path.join(script_dir, "strategy_names.json")
        with open(strategy_json_path, "r") as f:
            strategy_names = json.load(f)
        strategy_name = strategy_names[strategy_index]
        print(f"strategy_name: {strategy_name}")
        transformed_IR, num = apply_strategy_to_IR(IR, name_start_idx, strategy_name, num, input_output_name)
        print(f'num of transformed IRs: {num}')
        print(f'strategy {strategy_index}, num_list: {num_list}')
        print(f'transformed_IR:{transformed_IR}')
        # num_list.append(num)
        # print(f'strategy {strategy_index}, num_list: {num_list}')
                
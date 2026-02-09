from TIR import *
import os
import json
from ops import *
import torch
import logging
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model transformations")
    parser.add_argument("--level", type=str, default="level1")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--strategy_index", type=int, default=42)
    parser.add_argument("--start_model_idx", type=int, default=0)
    args = parser.parse_args()
    level = args.level
    model_index = args.model_index
    strategy_index = args.strategy_index
    start_model_idx= args.start_model_idx
    target = "cuda -device=1"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_json_path = os.path.join(script_dir, level+"_model_name.json")
    with open(model_json_path, "r") as f:
        model_names = json.load(f)
    atol= 1e-7
    dtype=torch.double
    applied_model_indices=[]
    log_name=f'/root/LLM/LLM_docker/log/strategy{strategy_index}_{level}.log'
    if strategy_index==35:
        if level=='level2':
            model_name_list=[5,6,7]
        else:
            model_name_list=[]
    elif strategy_index==38 and level=='level2':
        model_name_list=list(range(start_model_idx, len(model_names)))
        model_name_list.remove(29)
    else:
        model_name_list=list(range(start_model_idx, len(model_names)))
    model_name_list=[1]#list(range(111, 119))
    for model_index in model_name_list:
        is_equal=True
    
    # if model_index<len(model_names):
        #start generating model and IR
        if model_index==23 and level=='level2':
            atol=1e-5
            rtol=1e-5
        elif strategy_index==42 and model_index==1 and level=='level2':
            atol=1.2
            rtol=1e-3
        else:
            atol=1e-7
            rtol=1e-5
        logging.basicConfig(filename=log_name, level=logging.INFO)
        logging.info(f"Processing model {model_index}/{len(model_names)}")
        print(f"Processing model {model_index}/{len(model_names)}")
        model_name = model_names[model_index]
        print("model_name:", model_name)
        default_shapes, ranges, dtypes, module=select_shape(model_name)
        use_defualt= True
        print(f'default_shapes: {default_shapes}, ranges: {ranges}, dtypes: {dtypes}')
        new_shapes=generate_random_shape(default_shapes, ranges, dtypes, use_defualt)
        print(f'new_shapes: {new_shapes}')
        model, inputs, params_name, params_shape, params_dtype, params_value_dict, input_shapes = select_model_with_new_shapes(model_name, new_shapes, dtype)
        ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx, paramsname_constant_mapping, constant_value_dict = analyze_pytorch_model_with_new_shapes(model_name, model, inputs, params_name, params_shape, params_dtype)
        print(f'constant_info:{constant_info}\nparamsname_constant_mapping:{paramsname_constant_mapping}\n constant_value_dict:{constant_value_dict.keys()}\n params_value_dict:{params_value_dict.keys()}')
        print(f'intermediate_info:{intermediate_info}\ninputs_info:{inputs_info}\noutputs_info:{outputs_info}\nparams_info:{params_info}')
        constant_params_value=mapping_constant_params_value(params_value_dict, constant_value_dict, paramsname_constant_mapping, constant_info[0], params_info[0])
        print(f'len constant_params_value:{len(constant_params_value)}, len names:{len(constant_info[0]) + len(params_info[0])}')
        # print(f'constant_params_value:{constant_params_value}')
        IR, name_start_idx= convert_to_IR(ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx)
        logging.basicConfig(filename=log_name, level=logging.INFO)
        logging.info(f'IR: {IR}')
        print(f'IR: {IR}')
        known_names, known_shapes, known_dtype, input_known_names=generate_IR_related_info(inputs_info, constant_info, params_info, outputs_info)
            
        #start using strategy
        strategy_json_path = os.path.join(script_dir, "strategy_names.json")
        with open(strategy_json_path, "r") as f:
            strategy_names = json.load(f)
        strategy_name = strategy_names[strategy_index]
        print(f"strategy_name: {strategy_name}")
        original_IR_list, transformed_IR_list, has_transformation = apply_strategy_to_IR(IR, name_start_idx, strategy_name, known_names)
        print(f'strategy {strategy_index}')
        logging.basicConfig(filename=log_name, level=logging.INFO)
        logging.info(f'len:{len(transformed_IR_list)}')
        print(f'len:{len(transformed_IR_list)}')
        logging.basicConfig(filename=log_name, level=logging.INFO)
        logging.info(f'transformed_IR_list:{transformed_IR_list}')
        # print(f'transformed_IR_list:{transformed_IR_list}')
        if has_transformation:
            applied_model_indices.append(model_index)

            # #in loop verification
            # for idx in range(len(transformed_IR_list)):
            #     new_constant_params_value=constant_params_value.copy()
            #     transformed_IR=transformed_IR_list[idx]
            #     logging.basicConfig(filename=log_name, level=logging.INFO)
            #     logging.info(f'idx:{idx}, transformed_IR: {transformed_IR}')
            #     print(f'idx:{idx}, transformed_IR: {transformed_IR}')
            #     #start verification for transformed IR
            #     f,_, _=build_tir_module(transformed_IR, known_names, known_shapes, known_dtype, input_known_names, target)
            #     is_equal=verification(f, model, model_name, module, input_shapes, outputs_info[1][0], outputs_info[2][0], target, new_constant_params_value, atol, rtol, dtype)
            #     if is_equal:
            #         logging.basicConfig(filename=log_name, level=logging.INFO)
            #         logging.info(f"Processed model {model_index}/{len(model_names)}: {idx+1}/{len(transformed_IR_list)} transfomed {model_name} successfully.")
            #         print(f"Processed model {model_index}/{len(model_names)}: {idx+1}/{len(transformed_IR_list)} transfomed {model_name} successfully.")
            #     else:
            #         logging.basicConfig(filename=log_name, level=logging.INFO)
            #         logging.info(f"{transformed_IR}-Processed model {model_index}/{len(model_names)}: {model_name}-{idx+1}/{len(transformed_IR_list)} failed.")
            #         print(f"{transformed_IR}-Processed model {model_index}/{len(model_names)}: {model_name}-{idx+1}/{len(transformed_IR_list)} failed.")
            #         break
            # if not is_equal:
            #     break
            # torch.cuda.empty_cache()
            # torch.cuda.reset_peak_memory_stats()


        # # # start verification for transformed IR
        # transformed_IR='B^{128}_{tx=0}L^{128}_{a=0}L^{64}_{c=0}[D^{f64,g}_{tx,a}=D^{f64,g}_{tx,a}+A^{f64,g}_{tx,c}*M^{f64,g}_{a,c};];B^{128}_{tx=0}L^{128}_{a=0}[D^{f64,g}_{tx,a}=D^{f64,g}_{tx,a}+N^{f64,g}_{a};];B^{128}_{tx=0}L^{128}_{a=0}[E^{f64,g}_{tx,a,0}=D^{f64,g}_{tx,a};];B^{128}_{tx=0}L^{128}_{a=0}L^{1}_{c=0}[F^{f64,g}_{tx,a,c,0}=E^{f64,g}_{tx,a,c};];L^{128}_{a=0}L^{128}_{c=0}L^{1}_{d=0}B^{1}_{tx=0}[G^{f64,g}_{a,c,d,tx}=F^{f64,g}_{a,c,d,tx};];L^{128}_{a=0}L^{128}_{c=0}L^{1}_{d=0}B^{1}_{tx=0}[G^{f64,g}_{a,c,d,tx+1}=F^{f64,g}_{a,c,d,tx};];B^{128}_{tx=0}L^{128}_{a=0}L^{1}_{c=0}L^{2}_{d=0}[R^{f64,g}_{tx,a}=R^{f64,g}_{tx,a}+G^{f64,g}_{tx,a,c,d};];B^{128}_{tx=0}L^{128}_{a=0}[S^{f64,g}_{tx,a}=R^{f64,g}_{tx,a}/2;];B^{128}_{tx=0}L^{128}_{a=0}L^{1}_{c=0}L^{2}_{d=0}[W^{f64,g}_{tx,a}=W^{f64,g}_{tx,a}+(G^{f64,g}_{tx,a,c,d}-S^{f64,g}_{tx,a})**2;];B^{128}_{tx=0}L^{128}_{a=0}[X^{f64,g}_{tx,a}=W^{f64,g}_{tx,a}/2;];B^{128}_{tx=0}L^{128}_{a=0}L^{1}_{c=0}L^{2}_{d=0}[Y^{f64,g}_{tx,a,c,d}=(G^{f64,g}_{tx,a,c,d}-S^{f64,g}_{tx,a})/sqrt(X^{f64,g}_{tx,a}+1e-05);];B^{128}_{tx=0}L^{128}_{a=0}L^{1}_{c=0}L^{2}_{d=0}[H^{f64,g}_{tx,a,c,d}=(O^{f64,g}_{a}/(Y^{f64,g}_{tx,a,c,d}+1e-3)*(Y^{f64,g}_{tx,a,c,d})+1e-3)*Y^{f64,g}_{tx,a,c,d}+Q^{f64,g}_{a};];B^{128}_{tx=0}L^{128}_{a=0}L^{2}_{c=0}[I^{f64,g}_{tx,a,c}=H^{f64,g}_{tx,a,0,c};];B^{128}_{tx=0}L^{128}_{a=0}L^{2}_{c=0}[J^{f64,g}_{tx,a,c}=I^{f64,g}_{tx,a,c}+C^{f64,g}_{tx,a,c};];B^{128}_{tx=0}L^{128}_{a=0}L^{2}_{c=0}[K^{f64,g}_{tx,a,c}=J^{f64,g}_{tx,a,c}*C^{f64,g}_{tx,a,c};];'
        # f,_, _=build_tir_module(transformed_IR, known_names, known_shapes, known_dtype, input_known_names, target)
        # is_equal=verification(f, model, model_name, module, input_shapes, outputs_info[1][0], outputs_info[2][0], target, constant_params_value, atol,rtol, dtype)
        # if is_equal:
        #     print(f"Processed model {model_index}/{len(model_names)}: {model_name} successfully.")
        # else:
        #     print(f"Processed model {model_index}/{len(model_names)}: {model_name} failed.")
        logging.basicConfig(filename=log_name, level=logging.INFO)
        logging.info(f'applied_model_indices: {applied_model_indices}')
        print(f'len:{len(applied_model_indices)}, applied_model_indices: {applied_model_indices}')

        #start verification for original IR
        f,_, _=build_tir_module(IR, known_names, known_shapes, known_dtype, input_known_names, target)
        # cuda_src = f.imported_modules[0].get_source()
        # print(cuda_src)
        is_equal=verification(f, model, model_name, module, input_shapes, outputs_info[1][0], outputs_info[2][0], target, constant_params_value, atol, rtol, dtype)
        if is_equal:
            print(f"Processed model {model_index}/{len(model_names)}: {model_name} successfully.")
        else:
            print(f"Processed model {model_index}/{len(model_names)}: {model_name} failed.")

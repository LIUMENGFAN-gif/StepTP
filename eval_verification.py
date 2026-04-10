import gzip
import json
import os
import torch
from TIR import *
from ops import *
import argparse
import time
import logging
import pickle

def error_record(error_record_dir, idx, error_info):
    with open(error_record_dir, 'a', encoding='utf-8') as f:
        json.dump({'eval_idx':idx, 'error': error_info}, f)
        f.write('\n')

def initial_time_record(tvm_time_record_dir, idx, tvm_time, pytorch_time, atol, rtol, MSE, strategy):
    with open(tvm_time_record_dir, 'a', encoding='utf-8') as f:
        json.dump({'eval_idx':idx, 'tvm_time': tvm_time, 'pytorch_time': pytorch_time, 'atol': float(atol), 'rtol': float(rtol), 'MSE': float(MSE), 'strategy': strategy}, f)
        f.write('\n')

def obtain_store_dict(info):
    base_name = os.path.basename(info)
    model_name=base_name[:base_name.index('_original')]
    with gzip.open(info, "rb") as f:
        store_dict = pickle.load(f)
    return store_dict, model_name

def obtain_basic_info(store_dict):
    known_names=store_dict['known_names']
    input_known_names=store_dict['input_known_names']
    input_shapes=store_dict['input_shapes']
    dtype=getattr(torch,store_dict['dtype'].split('.')[-1]) 
    output_dtype=getattr(torch,store_dict['output_dtype'].split('.')[-1])
    known_dtype=[getattr(torch, item.split('.')[-1]) for item in store_dict['known_dtype']]
    known_shapes=[torch.Size(item) for item in store_dict['known_shapes']]
    output_shapes=torch.Size(store_dict['output_shape'])
    constant_value_dict={k: torch.tensor(v[0], dtype=getattr(torch, v[-1].split('.')[-1])) for k, v in store_dict['constant_value_dict'].items()}
    params_value_dict={k: torch.tensor(v[0], dtype=getattr(torch, v[-1].split('.')[-1])) for k, v in store_dict['params_value_dict'].items()}
    #then get the model info
    _, _, _, module=select_shape(model_name)
    new_model, _, _, _, _, _, _ = select_model_with_new_shapes(model_name, store_dict['shape'], dtype)
    new_model=mapping_params_to_new_model(new_model, params_value_dict)
    constant_params_value=mapping_constant_params_value(params_value_dict, constant_value_dict, store_dict['paramsname_constant_mapping'], store_dict['constant_names'], store_dict['params_names'])
    print("done")
    return known_names, known_shapes, known_dtype, input_known_names, new_model, model_name, module, input_shapes, output_shapes, output_dtype, constant_params_value, dtype

def judge_if_equivalent(transformed_IR, atol, rtol, MSE):
    if rtol<=5e-3 or atol<=1e-7 or MSE<=1e-5:
        return True
    if 'f16' in transformed_IR:
        if atol<1.5:
            return True
    elif 'f32' in transformed_IR:
        if atol<1e-3:
            return True
    return False

def process_one_answer(idx, store_dict, model_name, transformed_IR, cuda, error_record_dir):
    known_names, known_shapes, known_dtype, input_known_names, new_model, model_name, module, input_shapes, output_shapes, output_dtype, constant_params_value, dtype=obtain_basic_info(store_dict)
    #start verification for transformed IR
    target = "cuda -device="+str(cuda)
    atol=1e-7
    rtol=1e-5
    try:
        f,_, is_built=build_tir_module(transformed_IR, known_names, known_shapes, known_dtype, input_known_names, target)
        if is_built:
            print("NOTE:built successfully")
            try:
                torch_time, tvm_time, atol, rtol, MSE=collect_verification(f, new_model, model_name, module, input_shapes, output_shapes, output_dtype, target, constant_params_value, atol, rtol, dtype)
                print("NOTE:executed successfully")
                is_equivalent=judge_if_equivalent(transformed_IR, atol, rtol, MSE)
                if is_equivalent:
                    print(f"NOTE:equivalent successfully, the speedup is {torch_time/tvm_time:.8f}x, the tvm time is {tvm_time:.8f}ms.")
                    if tvm_time<torch_time:
                        print(f"NOTE:speedup successfully, the speedup is {torch_time/tvm_time:.8f}x")
                else:
                    if 'f16' in transformed_IR:
                        error_info=f"not equivalent, float16, atol {atol:.2e}, rtol {rtol:.2e}, MSE {MSE:.2e}."
                    elif 'f32' in transformed_IR:
                        error_info=f"not equivalent, float32, atol {atol:.2e}, rtol {rtol:.2e}, MSE {MSE:.2e}."
                    else:
                        error_info=f"not equivalent, float64, atol {atol:.2e}, rtol {rtol:.2e}, MSE {MSE:.2e}."
                    error_record(error_record_dir, idx, error_info)
            except Exception as e:
                error_info=f"execute fails:{e}"
                error_record(error_record_dir, idx, error_info)
        else:
            error_info=f"build_tir_module fails 1:{f}"
            error_record(error_record_dir, idx, error_info)
    except Exception as e:
        error_info=f"build_tir_module fails 2:{e}"
        error_record(error_record_dir, idx, error_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model transformations")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--transformed_IR", type=str)
    parser.add_argument("--info", type=str)
    parser.add_argument("--num_IRs", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument("--error_record_dir", type=str, default='../nfs_folder/data_entries/error_record_RL.json')
    args = parser.parse_args()
    # args.transformed_IR='B^{96}_{tx=0}L^{94}_{a=0}L^{30}_{c=0}L^{89}_{d=0}[F^{f16,g}_{tx,a}=F^{f16,g}_{tx,a}+A^{f16,g}_{tx,a,c,d};];B^{96}_{tx=0}L^{94}_{a=0}[G^{f16,g}_{tx,a}=F^{f16,g}_{tx,a}/2670;];B^{96}_{tx=0}L^{94}_{a=0}L^{30}_{c=0}L^{89}_{d=0}[H^{f16,g}_{tx,a}=H^{f16,g}_{tx,a}+(A^{f16,g}_{tx,a,c,d}-G^{f16,g}_{tx,a})**2;];B^{96}_{tx=0}L^{94}_{a=0}[I^{f16,g}_{tx,a}=H^{f16,g}_{tx,a}/2670;];B^{96}_{tx=0}L^{94}_{a=0}L^{30}_{c=0}L^{89}_{d=0}[C^{f16,g}_{tx,a,c,d}=D^{f16,g}_{a}*((A^{f16,g}_{tx,a,c,d}-G^{f16,g}_{tx,a})/sqrt(I^{f16,g}_{tx,a}+1e-05))+E^{f16,g}_{a};];'
    # args.idx=0 
    # args.info='../nfs_folder/dataset/InstanceNorm_original_59.pkl.gz'
    # args.num_IRs=1 
    # args.cuda=0 
    # args.error_record_dir='../nfs_folder/evaluation/once/error_record_Qwen_32B.json'
    store_dict, model_name=obtain_store_dict(args.info)
    process_one_answer(args.idx, store_dict, model_name, args.transformed_IR, args.cuda, args.error_record_dir)

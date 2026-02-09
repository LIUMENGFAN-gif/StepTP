import gzip
import json
import os
import torch
from TIR import *
from ops import *
import subprocess
import logging
import argparse
import ast

def error_record(error_record_dir, idx, error_info):
    with open(error_record_dir, 'a', encoding='utf-8') as f:
        json.dump({'eval_idx':idx, 'error': error_info}, f)
        f.write('\n')

def use_cuda_single(cuda_idx, idx, transformed_IR, info, answer_dict, num_IRs, eval_result, error_record_dir, tvm_time_record_dir, log_name, label_type):
    print("start subprocess")
    proc = subprocess.Popen(
            ['timeout', '5m', 'python', '/root/LLM/LLM_docker/eval_verification.py', '--idx', str(idx), '--transformed_IR', transformed_IR, '--info', info, '--applied_strategy', answer_dict['applied_strategy'], '--num_IRs', str(num_IRs), '--cuda', str(cuda_idx), '--error_record_dir', error_record_dir, '--tvm_time_record_dir', tvm_time_record_dir, '--log_name', log_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    stdout, stderr = proc.communicate()
    print("end subprocess")
    if "NOTE:built successfully" in stdout.decode():
        eval_result[label_type]['build_pass']+=1
    if "NOTE:executed successfully" in stdout.decode():
        eval_result[label_type]['execute_pass']+=1
    if "NOTE:equivalent successfully" in stdout.decode():
        eval_result[label_type]['equal_pass']+=1
    if "NOTE:speedup successfully" in stdout.decode():
        eval_result[label_type]['speedup_pass']+=1
    if proc.returncode == 124 or proc.returncode == 137:
        error_record(error_record_dir, idx, "execution timeout")
    elif proc.returncode == 0:
        print("normal termination")
    else:
        error_info=f"abnormal termination, returncode={proc.returncode}, stderr={stderr.decode()}"
        error_record(error_record_dir, idx, error_info)

def deal_with_string_answer_dict(answer_dict):
    answer_dict=answer_dict[:answer_dict.rfind('\"}')+2]+']'
    try:
        answer_dict=ast.literal_eval(answer_dict)
    except:
        answer_dict=answer_dict
    return answer_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model transformations")
    parser.add_argument("--cuda_idx", type=int, default=3)
    parser.add_argument("--machine", type=int, default=16)
    parser.add_argument("--has_rag", type=str, default="f")#n: no rag, f: full strategy, p: partial strategy
    args = parser.parse_args()
    cuda_idx=args.cuda_idx
    machine=args.machine
    has_rag=args.has_rag
    if machine==14:
        eval_data_file_name='eval_dataset'
        eval_results_folder='eval_results_g14'
    elif machine==16:
        eval_data_file_name='eval_dataset_100'
        eval_results_folder='eval_results'
    elif machine==17:
        eval_data_file_name='eval_dataset'
        eval_results_folder='eval_results_g17'
    if has_rag=='n':
        sign=''
    elif has_rag=='f':
        sign='_kb'
    elif has_rag=='p':
        sign='_kb_faiss'
    #---------------eval dataset--------------------
    eval_result_json_dir=f'/root/nfs_folder/data_entries/eval_result_g{machine}_{has_rag}.json'
    error_record_dir=f'/root/nfs_folder/data_entries/error_record_g{machine}_{has_rag}.json'
    tvm_time_record_dir=f'/root/nfs_folder/data_entries/tvm_time_record_g{machine}_{has_rag}.json'
    log_name=f'/root/LLM/LLM_docker/log/elog/eval_g{machine}_{has_rag}.log'
    eval_result={'single':{'total_num':0, 'different_pass':0, 'build_pass':0, 'execute_pass':0, 'equal_pass':0, 'speedup_pass':0},\
                    'multiple':{'total_prompt':0, 'total_num':0, 'require_num_pass':0, 'different_pass':0, 'build_pass':0, 'execute_pass':0, 'equal_pass':0, 'speedup_pass':0}}
    with gzip.open(f'/root/nfs_folder/data_entries/{eval_data_file_name}.json.gz', "rt") as f:
        eval_dataset = json.load(f)
    for idx in range(500):
        with open(f'/root/nfs_folder/data_entries/{eval_results_folder}/eval{sign}_{idx}.json', 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        query = eval_dataset[idx]['prompt']
        original_IR=query[query.index(':')+1:query.index(', where the known')].strip()
        info=eval_data['info']
        num_IRs=eval_data['num_IRs']
        answer_dict=eval_data['answer']
        print(f'answer_dict:{answer_dict}')
        print(f'type(answer_dict):{type(answer_dict)}')
        if isinstance(answer_dict, str):
            answer_dict=deal_with_string_answer_dict(answer_dict)
            print(f'after dealing string, type(answer_dict):{type(answer_dict)}')
        if num_IRs==1:
            eval_result['single']['total_num']+=1
            if isinstance(answer_dict, list):
                answer_dict=answer_dict[0]
            try:
                transformed_IR=answer_dict['transformed_IR']
                logging.basicConfig(filename=log_name, level=logging.INFO)
                logging.info(f"Processing eval{sign}_{idx}")
                print(f"Processing eval{sign}_{idx}")#: info:{info}, num_IRs:{num_IRs}, original_IR:{original_IR}, answer_dict:{answer_dict}")
                if original_IR!=transformed_IR:
                    eval_result['single']['different_pass']+=1
                    print("The original IR and transformed IR are different.")
                    use_cuda_single(cuda_idx, idx, transformed_IR, info, answer_dict, num_IRs, eval_result, error_record_dir, tvm_time_record_dir, log_name, 'single')
                else:
                    error_info='original IR and transformed IR are the same'
                    error_record(error_record_dir, idx, error_info)
            except:
                error_info='answer dict has wrong type.'
                error_record(error_record_dir, idx, error_info)
            print("main done")
        else:
            eval_result['multiple']['total_prompt']+=1
            logging.basicConfig(filename=log_name, level=logging.INFO)
            logging.info(f"Processing eval_{idx}")
            try:
                transformed_IR_list=[answer_dict_item['transformed_IR'] for answer_dict_item in answer_dict]
                print(f'transformed_IR_list before set:{len(transformed_IR_list)}')
                transformed_IR_list=list(set(transformed_IR_list))
                item_idx_list=[next((i for i, d in enumerate(answer_dict) if item in d.values()), -1) for item in transformed_IR_list]
                print(f'transformed_IR_list after set:{len(transformed_IR_list), len(item_idx_list), item_idx_list}')
                eval_result['multiple']['total_num']+=len(transformed_IR_list)
                if len(transformed_IR_list)==num_IRs:
                    eval_result['multiple']['require_num_pass']+=1
                for item_idx in item_idx_list:
                    answer_dict_item=answer_dict[item_idx]
                    transformed_IR=answer_dict_item['transformed_IR']
                    logging.basicConfig(filename=log_name, level=logging.INFO)
                    logging.info(f"Processing eval{sign}_{idx}: {item_idx}")
                    if original_IR!=transformed_IR:
                        eval_result['multiple']['different_pass']+=1
                        print("The original IR and transformed IR are different in multiple labels.")
                        use_cuda_single(cuda_idx, idx, transformed_IR, info, answer_dict_item, num_IRs, eval_result, error_record_dir, tvm_time_record_dir, log_name, 'multiple')
                    else:
                        error_info='original IR and transformed IR are the same'
                        error_record(error_record_dir, idx, error_info)
            except:
                error_info='answer dict has wrong type.'
                error_record(error_record_dir, idx, error_info)
        logging.basicConfig(filename=log_name, level=logging.INFO)
        logging.info(f'eval_result:{eval_result}')
    print(f'eval_result:{eval_result}')
    with open(eval_result_json_dir, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f)
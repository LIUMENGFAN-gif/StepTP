from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template
import gzip
import json
import os
import torch
import tqdm
import ast
import argparse


def infer(engine, infer_request):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
    resp_list = engine.infer(infer_request, request_config)
    response = [resp.choices[0].message.content for resp in resp_list]
    return response

def batch_infer(engine, start_idx, end_idx):
  infer_requests=[InferRequest(messages=[{'role': 'user', 'content': eval_dataset[i]['prompt']}]) for i in range(start_idx, end_idx)]
  return infer_requests

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model transformations")
    parser.add_argument("--cuda", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    # generation_config
    max_new_tokens = 40960
    temperature = 0
    #---------------eval dataset--------------------
    with gzip.open(f'../nfs_folder/data_entries/eval_dataset_updated.json.gz', "rt") as f:
        eval_dataset = json.load(f)
    print(f'eval_dataset length: {len(eval_dataset)}')
    # print(f'eval_dataset[0]:{eval_dataset[0]}')
    #--------------model downloading-----------------
    # Hyperparameters for inference
    # model_size="Qwen_1_7B"
    # output_dir="../nfs_folder/training/sft_1_7B_CoT_filtered_v2"
    # last_model_checkpoint = output_dir+'/checkpoint-454'
    # model_id_or_path='../nfs_folder/LLM/Qwen3-1.7B/stage_1_model'
    # model_size="Qwen_8B"
    # output_dir="../nfs_folder/training/sft_8B_CoT_filtered_v2"
    # last_model_checkpoint = output_dir+'/checkpoint-1812'
    # model_id_or_path='../nfs_folder/LLM/Qwen3-8B/stage_1_model'
    # model_size="Qwen_14B"
    # output_dir="../nfs_folder/training/sft_14B_CoT_filtered_v2"
    # last_model_checkpoint = output_dir+'/checkpoint-906'
    # model_id_or_path='../nfs_folder/LLM/Qwen3-14B/stage_1_model'
    model_size="Qwen_32B"
    output_dir="../nfs_folder/training/sft_32B_CoT_filtered_v2"
    last_model_checkpoint = output_dir+'/checkpoint-1812'
    model_id_or_path='../nfs_folder/LLM/Qwen3-32B/stage_1_model'
    system = 'You are a helpful assistant.'
    infer_backend = 'pt'
    stream = True
    # Get model and template, and load LoRA weights.
    engine = PtEngine(model_id_or_path, model_type="qwen3", adapters=[last_model_checkpoint], max_batch_size=args.batch_size)
    template = get_template(engine.model_meta.template, engine.processor, default_system=system)
    # You can modify the `default_template` directly here, or pass it in during `engine.infer`.
    engine.default_template = template
    print("downloaded model and lora.")
    #inference:
    start=0
    end=int(len(eval_dataset))
    for start_idx in tqdm.tqdm(range(start, end, args.batch_size)):
      end_idx=min(start_idx+args.batch_size, end)
      infer_requests=batch_infer(engine, start_idx, end_idx)
      answer_list = infer(engine, infer_requests)
      for idx in range(min(args.batch_size, end_idx-start_idx)):
        eval_results=str({'answer':answer_list[idx], 'info': eval_dataset[idx+start_idx]['info']})
        with open(f'../nfs_folder/evaluation/once/{model_size}/eval_{idx+start_idx}.json', 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=4)


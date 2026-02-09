import math
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template
import gzip
import json
import os
import math
import tqdm
import ast
import argparse
import subprocess
import re
from transformers import set_seed
from collections import deque

def error_record(error_record_dir, idx, error_info):
    with open(error_record_dir, 'a', encoding='utf-8') as f:
        json.dump({'eval_idx':idx, 'error': error_info}, f)
        f.write('\n')

def use_cuda_single(cuda_idx, idx, transformed_IR, info, num_IRs, verification_result, error_record_dir):
    print("start subprocess")
    proc = subprocess.Popen(
            ['timeout', '6m', 'python', '../LLM_backup_g16_RL/LLM_docker/eval_verification.py', '--idx', str(idx), '--transformed_IR', transformed_IR, '--info', info, '--num_IRs', str(num_IRs), '--cuda', str(cuda_idx), '--error_record_dir', error_record_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    print("proc begins...")
    stdout, stderr = proc.communicate()
    stdout_context=stdout.decode()
    print("end subprocess")
    if "NOTE:built successfully" in stdout_context:
        verification_result['build_pass']+=1
    if "NOTE:executed successfully" in stdout_context:
        verification_result['execute_pass']+=1
    if "NOTE:equivalent successfully" in stdout_context:
        verification_result['equal_pass']+=1
        speedup_index=stdout_context.index("the speedup is")
        speedup_context=stdout_context[speedup_index+len("the speedup is")+1:]
        speedup_value=float(speedup_context[:speedup_context.index("x")])
        tvm_time=float(speedup_context[speedup_context.index("the tvm time is ")+len("the tvm time is "):speedup_context.index("ms")])
        verification_result['speedup'].append(speedup_value)
        verification_result['tvm_time'].append(tvm_time)
    if "NOTE:speedup successfully" in stdout_context:
        verification_result['speedup_pass']+=1
    if proc.returncode == 124 or proc.returncode == 137:
        error_record(error_record_dir, idx, "execution timeout")
    elif proc.returncode == 0:
        print("normal termination")
    else:
        error_info=f"abnormal termination, returncode={proc.returncode}, stderr={stderr.decode()}"
        error_record(error_record_dir, idx, error_info)
    # print(stdout_context)
    return verification_result

def infer(engine: InferEngine, infer_request: InferRequest, temperature):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
    resp_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = resp_list[0].choices[0].message.content
    return response

def try_answer_string(answer):
    try:
        ast.literal_eval(answer)
    except:
        try:
            answer=deal_with_exceed_long_answer(answer)
            ast.literal_eval(answer)
        except:
            answer=answer
    return answer

def deal_with_exceed_long_answer(answer):
    if answer[0]=='{':
        if 'idx' in answer:
            print("deal_with_exceed_long_answer idx in answer")
            answer='['+answer+']'
            answer=try_answer_string(answer)
            return answer
        else:
            print("deal_with_exceed_long_answer no idx in answer")
            return '{}'
    elif answer[0]=='[':
        if '}' in answer:
            print("deal_with_exceed_long_answer } in answer")
            answer=answer[:answer.rfind('\"}')+2]+']'
            return answer
        else:
            print("deal_with_exceed_long_answer no } in answer")
            return '[]'
    elif '</think>' in answer:
        print("deal_with_exceed_long_answer think in answer")
        answer=answer[answer.rfind('</think>')+len('</think>'):].strip()
        print(f"deal after:{answer}")
        if "```json" in answer:
            answer=answer.replace("```json","").replace("```","")
        # print(f'deal_with_exceed_long_answer answer after think:{answer}')
        answer=try_answer_string(answer)
        # print(f'deal_with_exceed_long_answer answer after recursion:{answer}')
        return answer
    else:
        return answer

def obtain_answer_dict(answer):
  if '<answer>' in answer and '<\\answer>' in answer:
    answer=re.search(r'<answer>(.*?)<\\answer>', answer, re.DOTALL).group(1).strip()
  elif '<answer>' in answer and '</answer>' in answer:
    answer=re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL).group(1).strip()
  try:
    answer_dict=ast.literal_eval(answer)
  except:
      try:
          answer=deal_with_exceed_long_answer(answer)
          answer_dict=ast.literal_eval(answer)
      except:
          answer_dict=answer
  return answer_dict

def construct_prompt(node, num_IRs, step_by_step_prompt, start_description, basic_description, requirement, strategy_prompt, hardware_info):
    current_IR=node.state
    depth=node.depth
    if node.different_original:
        current_applied_strategy=node.state_dict['applied_strategy']
        current_info="The speedup value of the current IR: "+str(node.speedup)+", the depth is "+str(depth)+", and the current IR is obtained from the parent IR using the strategy \'"+current_applied_strategy+"\'"
    else:
        current_info="The current IR is the same as the root IR, and the depth is "+str(depth)
    if node.trials>0:
        current_info+=". This current IR has been transformed "+str(node.trials)+" times but failed to pass the verification."
        if not node.different_original:
            current_info+=f"Last time, the current IR was transformed to the root IR as \'{node.failed_children[-1][0]}\'.\n"
        elif not node.different:
            current_info+="Last time, the current IR keeped the same after transformation.\n"
        else:
            current_info+="Last time, "
            for idx in range(num_IRs):
                if node.failed_children[-idx][1]['different_original']==0:
                    current_info+=f"the transformed IR {idx}, \'{node.failed_children[-idx][0]}\', was same as the root IR"
                elif node.failed_children[-idx][1]['different_original']==1:
                    current_info+=f"the transformed IR {idx}, \'{node.failed_children[-idx][0]}\', was same as the current IR"
                else:
                    current_info+=f"the transformed IR {idx}, \'{node.failed_children[-idx][0]}\', was not numerically equivalent to the original IR"
                if idx==num_IRs-1:
                    current_info+=". You MUST compare with these failed transformed IRs when generating new IR. You are FORBIDDEN to generate these failed transformed IRs again. \n"
                else:
                    current_info+="; "
    else:
        current_info+='.\n'
    parent_info=""
    if node.parent:
        if node.different:
            if not node.parent.equal:
                parent_info="History:\nThe parent IR: \'"+str(node.parent.state_dict)+"\', depth:"+str(node.parent.depth)+", non-equal IR.\n"
            elif 'original_IR' in str(node.parent.state_dict):
                parent_info="History:\nThe parent IR: same as the root IR, depth:0, speedup value: 1. \n"
            else:
                parent_info="History:\nThe parent IR: \'"+str(node.parent.state_dict)+"\', depth:"+str(node.parent.depth)+", speedup value: "+str(node.parent.speedup)+'\n'
        else:
            parent_info="The parent IR: same as the current IR. \n"
    return step_by_step_prompt+"\n"+start_description+": \'"+current_IR+"\', "+basic_description+parent_info+current_info+hardware_info+strategy_prompt+requirement

def obtain_node_parent(node):
    node_history=[]
    while node:
        node_history=[node.state_dict]+node_history
        node = node.parent
    return node_history

class BFSNode:
    def __init__(self, state, state_dict={}, depth=0, parent=None):
        self.state = state  # string
        self.state_dict=state_dict
        self.parent = parent
        self.children = []  
        self.failed_children=[]
        self.speedup = 1
        self.equal = True
        self.different = True
        self.different_original = False
        self.trials=0
        self.depth=depth

class BFS:
    def __init__(self, max_depth, info, step_by_step_prompt, hardware_info, strategy_prompt, num_IRs, start_description, basic_description, requirement, original_tvm_time, temperature, engine, veri_cuda, eval_idx, database, error_record_dir):
        self.max_depth = max_depth
        self.info = info
        self.step_by_step_prompt = step_by_step_prompt
        self.hardware_info = hardware_info
        self.strategy_prompt = strategy_prompt
        self.num_IRs = num_IRs
        self.start_description = start_description
        self.basic_description = basic_description
        self.requirement = requirement
        self.database = database
        self.original_tvm_time = sum(original_tvm_time)/len(original_tvm_time)
        self.temperature = temperature
        self.engine = engine
        self.veri_cuda = veri_cuda
        self.eval_idx = eval_idx
        self.error_record_dir = error_record_dir
        self.speedup_dict={}
        self.repeat_num=0

    def search(self, original_IR):
        self.original_IR=original_IR
        root = BFSNode(state=original_IR, state_dict={"original_IR": original_IR})
        #-------------create a queue for BFS-----------------
        queue = deque([root])
        while queue:
            current_node = queue.popleft()
            #---------------check terminal condition-----------------
            if self._is_terminal(current_node):
                continue
            #---------------expand the current node----------------
            target_node_list, IR_in_prompt = self._expand(current_node)
            print(f'len(target_node_list):{len(target_node_list)}')
            #---------------------simulation----------------
            all_verification_result_list = self._simulate(target_node_list, current_node.state)
            #---------------------check_children-------------
            self._check_children(current_node, target_node_list, all_verification_result_list)

            #--------------add to queue--------------------------
            for new_node in reversed(target_node_list):
                queue.append(new_node)
        try:
            final_IR=max(self.speedup_dict, key=lambda k: self.speedup_dict[k][0])
            return final_IR, self.speedup_dict[final_IR][0], self.speedup_dict[final_IR][1], self.database
        except:
            return "", 0, root, self.database
    
    def _expand(self, node):
        prompt=construct_prompt(node, self.num_IRs, self.step_by_step_prompt, self.start_description, self.basic_description, self.requirement, self.strategy_prompt, self.hardware_info)
        node.trials+=1
        IR_in_prompt=node.state
        print(f'prompt:{prompt}')
        answer = infer(self.engine, InferRequest(messages=[{'role': 'user', 'content': prompt}]), self.temperature)
        print(f'answer:{answer}')
        answer_list=obtain_answer_dict(answer)
        target_node_list=[]
        for answer_dict in answer_list:
            try:
                transformed_IR=answer_dict['transformed_IR']
            except:
                transformed_IR=answer
            target_node_list.append(BFSNode(state=transformed_IR, state_dict=answer_dict, depth=node.depth+1))
        return target_node_list, IR_in_prompt
    
    def _check_children(self, current_node, target_node_list, all_verification_result_list):
        for idx, verification_result in enumerate(all_verification_result_list):
            target_node=target_node_list[idx]
            if verification_result['different_original']>0 and verification_result['different_pass']>0 and verification_result['equal_pass']>0:
                avg_tvm=sum(verification_result['tvm_time'])/len(verification_result['tvm_time'])
                target_node.speedup=self.original_tvm_time/avg_tvm
                target_node.parent=current_node
                target_node.different_original=True
                target_node.different=True
                current_node.children.append(target_node)
                self.speedup_dict[target_node.state]=[target_node.speedup, target_node]
            else:
                current_node.failed_children.append((target_node.state_dict, verification_result))

    def _simulate(self, node_list, IR_in_prompt):
        all_verification_result_list=[]
        for node in node_list:
            transformed_IR=node.state
            if transformed_IR not in self.database:
                self.repeat_num=0
                verification_result={'different_original':0, 'different_pass':0, 'build_pass':0, 'execute_pass':0, 'equal_pass':0, 'speedup_pass':0, 'speedup': [], 'tvm_time':[]}
                try:
                    verification_result=use_cuda_single(self.veri_cuda, self.eval_idx, transformed_IR, self.info, self.num_IRs, verification_result, self.error_record_dir)
                except Exception as e:
                    error_record(self.error_record_dir, self.eval_idx, e)
                self.database[transformed_IR]=verification_result
            else:
                self.repeat_num+=1
                verification_result=self.database[transformed_IR]
            if transformed_IR!=self.original_IR:
                verification_result['different_original']=1
            else:
                verification_result['different_original']=0
            if transformed_IR!=IR_in_prompt:
                verification_result['different_pass']=1
            else:
                verification_result['different_pass']=0
            all_verification_result_list.append(verification_result)
            print(f'verification_result:{verification_result}')
        return all_verification_result_list
    
    def _is_terminal(self, current_node):
        """
        whether this state is a terminal state
        """
        if current_node.depth >= self.max_depth:
            return True
        elif current_node.trials > 3:
            return True
        else:
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model transformations")
    parser.add_argument("--cuda", type=str, default="0,1")
    parser.add_argument("--GPU", type=str, default="H20-3e")
    parser.add_argument("--veri_cuda", type=str, default="0")
    parser.add_argument("--start_data_idx", type=int, default=0)
    parser.add_argument("--end_data_idx", type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # generation_config
    max_new_tokens = 40960
    temperature = 0.3
    num_IRs=2
    max_depth=4
    set_seed(42)
    #---------------eval dataset--------------------
    with open(f'../nfs_folder/data_entries/eval_dataset_updated_equal_pass_filtered.json', "r") as f:
        eval_dataset = json.load(f)
    print(f'eval_dataset length: {len(eval_dataset)}')
    #--------------model downloading-----------------
    # Hyperparameters for inference
    model_size="Qwen_32B"
    output_dir="../nfs_folder/training/sft_32B_CoT_filtered_v2"
    last_model_checkpoint = output_dir+'/checkpoint-1812'
    model_id_or_path='../nfs_folder/LLM/Qwen3-32B/stage_1_model'
    system = 'You are a helpful IR optimization assistant.'
    infer_backend = 'pt'
    stream = True
    engine = PtEngine(model_id_or_path, model_type="qwen3", adapters=[last_model_checkpoint])
    template = get_template(engine.model_meta.template, engine.processor, default_system=system)
    engine.default_template = template
    print("downloaded model and lora.")
    #--------------pre-settings----------------------------
    step_by_step_prompt="Breadth-first-based optimization is used on a given IR to improve performance. Each IR is a state, and has a parent transformation and speedup performance."
    hardware_info=f"**Target hardware**: NVIDIA {args.GPU} GPU.\nCUDA binding rules: Loop axes bound to block (along x,y,z axis, max dimension value: 2^31-1, 65535, 65535) MUST be renamed with prefixes bx, by, bz and unique, respectively, followed by other unique lowercase letters. Loop axes bound to thread (along x,y,z axis, max dimension value: 1024, 1024, 64) MUST be renamed with prefixes tx, ty, tz, respectively, followed by other unique lowercase letters.\n"
    hardware_info+="Memory usage rules: Data indexed by block-level loops may be placed in shared (s) or global (g) memory. Data indexed by thread-level loops may be placed in local (l), shared (s) or global (g) memory.\n"
    strategy_name="operator fusion, operator fission, compute inline, expression splitting, tensor concat to fuse operators, tensor split to decouple operators, common subexpression elimination, expression reorder, loop reorder, loop tiling, loop split, loop fusion, loop unrolling, loop parallelization, loop vectorization, loop binding, reduction factorization, cache read write, layout transformation, set storage scope, set storage layout, precompute indices, factorization, expand factorization, cancellation,expand cancellation, apart, together, powsimp, expand powsimp, expand log, logsimp, collect, expand collect, partially equivalent then correct, normal loop max to prefix max, exponential split, multiplicative split, additive split, normal loop summation on exp to prefix summation on exp, online softmax, flashattention wo tiling, normal matmul to prefix matmul based on online softmax"
    strategy_prompt="The following strategies and any other mathematical strategies can be considered:\n"+strategy_name+".\n"
    error_record_dir=f'../nfs_folder/evaluation/step_by_step/BFS_based/{args.GPU}/error_record_{model_size}.json'
    # os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda}, {args.veri_cuda}'
    #---------------eval data------------------------------
    for idx in tqdm.tqdm(range(args.start_data_idx, args.end_data_idx)):
        # print(f'eval_dataset[idx]:{eval_dataset[idx]}')
        #----------------data preparation-------------------
        info=eval_dataset[idx]['info']
        prompt = eval_dataset[idx]['prompt']
        original_IR = eval_dataset[idx]['original_IR']
        original_tvm_time=eval_dataset[idx]['tvm_time']
        original_speedup=eval_dataset[idx]['pytorch_speedup']
        database={original_IR:{'eval_idx':idx, 'different_original': 0, 'different_pass':0, 'build_pass':1, 'execute_pass':1, 'equal_pass':1, 'speedup_pass':0, 'speedup': original_speedup, 'tvm_time':original_tvm_time}}
        #----------------split prompt-----------------------
        start_description=prompt[:prompt.index(':')].replace("following", "current")
        basic_description=prompt[prompt.index('where the known variables'):prompt.index('Please give me a **numerically equivalent transformed** IR')]
        requirement=prompt[prompt.index('Please give me a **numerically equivalent transformed** IR'):].replace('Please give me a **numerically equivalent transformed** IR that produces', f'Please give me at least {num_IRs} different **numerically equivalent transformed** IRs that produce').replace("and also provide applied strategy in this transformed IR","and also provide applied strategy for each transformed IR")
        requirement=requirement.replace("numerically equivalent transformed", "numerically equivalent, runtime-performance-optimized").replace("(bitwise identical),", "(bitwise identical) to achieve higher speedup values (should be more than 1),")
        requirement=requirement.replace("Return the answer **only** as a valid JSON object with the following keys:\'transformed_IR\', \'applied_strategies\'.", "Return the answer list **only** as a valid JSON object, and each entry with the following keys: \'idx\', \'transformed_IR\', \'applied_strategies\'.")
        requirement="**Task**:\n"+requirement
        requirement+="\nCRITICAL:\n1. Before you suggest each new transformation, you MUST identify what has been changed in the current IR compared to the root IR. For each new optimization, you MUST build it ON TOP OF these existing changes, namely ON TOP OF the current IR. The strategies MUST be used on the current IR! You MUST compare your modified parts in each transformed IR with the current IR. If they are identical strings, your answer is WRONG. If the unmodified part in each transformed IR is different from the current IR, your answer is WRONG.\n2.Don't repeat the current or parent IRs! You MUST NOT revert to the parent IR: In particular, you are NOT allowed to apply any reverse or undo operation that reconstructs the current IR from its parent IR, including inverse transformations such as operator fusion <-> operator fission, loop tiling <-> loop fusion, loop split <-> loop fusion, apart <-> together, collect <-> expand collect, or similar reversals."
        # print(f'requirement:{requirement}')
        # print("done")
        dfs=BFS(max_depth, info, step_by_step_prompt, hardware_info, strategy_prompt, num_IRs, start_description, basic_description, requirement, original_tvm_time, temperature, engine, 0, idx, database, error_record_dir)
        print("BFS created")
        optimized_IR, speedup, final_node, database=dfs.search(original_IR)
        node_history = obtain_node_parent(final_node)
        print(f'optimized_IR:{optimized_IR}, speedup:{speedup}')
        print(f'node_history:{node_history}')
        with open(f'../nfs_folder/evaluation/step_by_step/BFS_based/{args.GPU}/result/result_{idx}.json', 'w') as f:
            json.dump({'optimized_IR':optimized_IR, 'speedup':speedup}, f)
        with open(f'../nfs_folder/evaluation/step_by_step/BFS_based/{args.GPU}/database/database_{idx}.json', 'w') as f:
            json.dump(database, f)
        with open(f'../nfs_folder/evaluation/step_by_step/BFS_based/{args.GPU}/node_history/history_{idx}.json', 'w') as f:
            json.dump(node_history, f)
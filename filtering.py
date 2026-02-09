import json
import tqdm
import ast
import difflib
import random

def check_strategy_statistic(train_dataset):
  #check the statistic
  labels=[item['label'] for item in train_dataset]
  print(f'len(labels):{len(labels)}')
  #obtain the statistic of strategy number
  strategy_dict={}
  single_strategy_dict={}
  for label in tqdm.tqdm(labels):
    label=ast.literal_eval(label)
    if isinstance(label, list):
      for label_item in label:
        applied_strategy_value=label_item["applied_strategy"]
        if applied_strategy_value in strategy_dict:
          strategy_dict[applied_strategy_value]+=1
        else:
          strategy_dict[applied_strategy_value]=1
    else:
      applied_strategy_value=label["applied_strategy"]
      if applied_strategy_value in strategy_dict:
        strategy_dict[applied_strategy_value]+=1
      else:
        strategy_dict[applied_strategy_value]=1
      if applied_strategy_value in single_strategy_dict:
        single_strategy_dict[applied_strategy_value]+=1
      else:
        single_strategy_dict[applied_strategy_value]=1
  print(f"strategy_dict:{strategy_dict}, sum:{sum(strategy_dict.values())}\n")
  print(f"single_strategy_dict:{single_strategy_dict}, sum:{sum(single_strategy_dict.values())}")

def filter_simple_data(dataset_with_CoT):
  new_dataset=[]
  num_multiple=0
  multi_strategy_dict={}
  strategy_num={"operator_fusion":0,"compute_inline":0, "expression_splitting":0, "expression_reorder":0, "loop_reorder":0, "loop_unrolling":0, "loop_parallelization":0, "loop_vectorization":0, "loop_binding":0, "set_storage_layout":0, "exponential_split":0, "multiplicative_split":0, "additive_split":0}
  single_strategy_num={"operator_fission":0, "factorization":0, "expand_factorization":0,"cancellation":0,"expand_cancellation":0,"apart":0,"together":0,"powsimp":0,"expand_powsimp":0,"logsimp":0,"expand_log":0,"collect":0,"expand_collect":0}
  for data_idx, data in tqdm.tqdm(enumerate(dataset_with_CoT)):
    label=ast.literal_eval(data['label'])
    original_IR=data['original_IR']
    data["label_with_CoT"]=data["label_with_CoT"].replace("<\\think>", "</think>").replace("<\\answer>", "</answer>")
    if isinstance(label, list):
      # print("before:",data, "\n")
      CoT_list=ast.literal_eval(data["CoT"])
      new_label=[]
      new_CoT_list=[]
      # old_num, new_num=len(label), len(label)
      for label_idx, label_item in enumerate(label):
        if label_item['applied_strategy'] not in ["operator_fission", "factorization", "expand_factorization","cancellation","expand_cancellation","apart","together","powsimp","expand_powsimp","logsimp","expand_log","collect","expand_collect"]:
          random_num=random.random()
          if label_item['applied_strategy'] not in ["operator_fusion","compute_inline", "expression_splitting", "expression_reorder", "loop_reorder", "loop_unrolling", "loop_parallelization", "loop_vectorization", "loop_binding", "set_storage_layout", "exponential_split", "multiplicative_split", "additive_split"]:
            new_label.append(label_item)
            new_CoT_list.append(CoT_list[label_idx])
          elif random_num>0.5 and strategy_num[label_item['applied_strategy']]<2000:
            new_label.append(label_item)
            new_CoT_list.append(CoT_list[label_idx])
            strategy_num[label_item['applied_strategy']]+=1
      if len(new_label)>1:
        tmp_transformed_sorted_list=sorted([new_label_item['transformed_IR'] for new_label_item in new_label])
        add_multiple_data=False
        if original_IR in multi_strategy_dict:
          if tmp_transformed_sorted_list in multi_strategy_dict[original_IR]:
            add_multiple_data=False
          else:
            multi_strategy_dict[original_IR].append(tmp_transformed_sorted_list)
            add_multiple_data=True
        else:
          multi_strategy_dict[original_IR]=[tmp_transformed_sorted_list]
          add_multiple_data=True
        # new_num=len(new_label)
        if add_multiple_data and "" not in new_CoT_list and len(new_label)==len(new_CoT_list):
          TIR_label=ast.literal_eval(str(data['TIR_label']))
          new_TIR_label=[]
          for idx, new_label_item in enumerate(new_label):
            new_TIR_label.append(TIR_label[new_label_item['idx']])
            new_label[idx]['idx']=idx
            new_TIR_label[idx]['idx']=idx
          if len(new_TIR_label)==len(new_label):
            data['prompt']=data['prompt'].replace(f"at least {len(label)}", f"at least {len(new_label)}")
            data["label"]=str(new_label)
            data["TIR_label"]=str(new_TIR_label)
            data["CoT"]=str(new_CoT_list)
            label_with_CoT=f'<think>These {len(new_label)} transformed IRs can be individually analyzed as follows: '
            for label_idx, label_item in enumerate(new_label):
              if new_CoT_list[label_idx]!="":
                label_with_CoT+=f'{label_idx}. '+new_CoT_list[label_idx]+'\n'
            label_with_CoT+='</think><answer>'+data['label']+'</answer>'
            data["label_with_CoT"]=label_with_CoT
            new_dataset.append(data)
            num_multiple+=1
      # print("after:",data)
      # if new_num<old_num:
      #   print(f"old_num:{old_num}, new_num:{new_num}")
      #   break
    else:
      random_num=random.random()
      if label['applied_strategy'] not in ["operator_fission", "factorization", "expand_factorization","cancellation","expand_cancellation","apart","together","powsimp","expand_powsimp","logsimp","expand_log","collect","expand_collect"]:
        new_dataset.append(data)
      elif random_num>0.5 and single_strategy_num[label['applied_strategy']]<50 and label['applied_strategy'] not in ["expand_factorization","expand_cancellation","apart","together","expand_powsimp","expand_log","expand_collect"]:
        new_dataset.append(data)
        single_strategy_num[label['applied_strategy']]+=1
      elif random_num>0.7 and single_strategy_num[label['applied_strategy']]<10:
        new_dataset.append(data)
        single_strategy_num[label['applied_strategy']]+=1
        
  print(f"len(new_dataset):{len(new_dataset)}, num_multiple:{num_multiple}")
  return new_dataset



if __name__ == '__main__':
  with open("../nfs_folder/data_entries/multi_IRs_train_dataset_filtered_with_CoT.jsonl", "r") as f:
    train_dataset=[json.loads(line) for line in tqdm.tqdm(f)]
  check_strategy_statistic(train_dataset)
  new_dataset=filter_simple_data(train_dataset)
  

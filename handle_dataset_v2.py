import gzip
import json
import os
import logging
import argparse
import random

def save_dataset(name,dataset):
    with gzip.open(f'../LLM/LLM_docker/dataset/dataset_entries/new/{name}.json.gz', "wt") as f:
        json.dump(dataset, f)
    logging.basicConfig(filename=log_name, level=logging.INFO)
    logging.info(f"collected. dataset:{len(dataset)}")

def obtain_collected_num(level, model_index):
    if level=='level2' and model_index==14:
        collected_num=20*27*3
    elif level=='level2' and model_index in [5,6,7]:
        collected_num=20*9*3
    elif (level=='level1' and model_index in [79,81]) or (level=='level2' and model_index in [2,3,4,8,9,10,12,13,15,16,17]):
        collected_num=20*3*2
    elif level=='level1' and model_index==40:
        collected_num=20*2
    else:
        collected_num=20*3
    return collected_num

def obtain_known_info(known_names, known_dtype, known_shapes):
    known_info=''
    for var_idx in range(len(known_names)):
        if var_idx<len(known_names)-1:
            known_info+=f'\'{known_names[var_idx]}\' with the dtype {known_dtype[var_idx]} and shape {known_shapes[var_idx]}, '
        else:
            known_info+=f'and \'{known_names[var_idx]}\' with the dtype {known_dtype[var_idx]} and shape {known_shapes[var_idx]}'
    return known_info

def construct_multiple_lable_sentence(full_store_dict_name, model_dataset, multiple_labels, model_name, original_IR, known_info, log_name, dataset_entry_idx):
    multi_dataset=[]
    multi_prompt=[]
    prompt_list=[]
    #construct multi dataset
    if len(multiple_labels)>1:
        selected_range=list(range(2,len(multiple_labels)+1))
        repeated_num=random.choice(selected_range)
        repeated_list=[]
        for _ in range(repeated_num):
            dataset_sentence={'prompt':'', 'label':'', 'info': full_store_dict_name}
            selected_num=random.choice(selected_range)
            selected_labels=random.sample(multiple_labels, selected_num)
            if selected_labels not in repeated_list:
                repeated_list.append(selected_labels)
                prompt=f'Give the following IR of {model_name}: \'{original_IR}\', where the known variables are {known_info}. Do not change the names, shapes or dtypes of these known variables in the IR. \nPlease give me at least {selected_num} different **numerically equivalent transformed** IRs that produces exactly the same outputs for any floating-point inputs (bitwise identical), and also provide applied strategy for each transformed IR. \n Return the answer list **only** as a valid JSON object, and each entry with the following keys: \'idx\', \'transformed_IR\', \'applied_strategies\'.'
                labels=[{'idx': i, **label} for i, label in enumerate(selected_labels)]
                dataset_sentence['prompt']=prompt
                dataset_sentence['label']=json.dumps(labels)
                model_dataset.append(dataset_sentence)
                multi_dataset.append(dataset_sentence)
                prompt_list.append(prompt)
                dataset_entry_idx+=1
                logging.basicConfig(filename=log_name, level=logging.INFO)
                logging.info(f"idx:{dataset_entry_idx}, multiple:{selected_num}, data:{dataset_sentence}")
                # print(f"idx:{dataset_entry_idx}, multiple:{selected_num}, data:{dataset_sentence}")
    #construct multi prompt:
    selected_range=list(range(2,20))
    repeated_num=random.choice(selected_range)
    repeated_list=[]
    for _ in range(repeated_num):
        selected_num=random.choice(selected_range)
        if selected_num not in repeated_list:
            dataset_sentence={'prompt':'', 'info': full_store_dict_name}
            repeated_list.append(selected_num)
            prompt=f'Give the following IR of {model_name}: \'{original_IR}\', where the known variables are {known_info}. Do not change the names, shapes or dtypes of these known variables in the IR. \nPlease give me at least {selected_num} different **numerically equivalent transformed** IRs that produces exactly the same outputs for any floating-point inputs (bitwise identical), and also provide applied strategy for each transformed IR. \n Return the answer list **only** as a valid JSON object, and each entry with the following keys: \'idx\', \'transformed_IR\', \'applied_strategies\'.'
            if prompt not in prompt_list:
                dataset_sentence['prompt']=prompt
                multi_prompt.append(dataset_sentence)
    return dataset_entry_idx, model_dataset,multi_dataset, multi_prompt

def obtain_store_dict_and_known_shapes(model_name, collected_idx):
    store_dict_name=f'{model_name}_original_{collected_idx}'
    full_store_dict_name=os.path.join(f'../nfs_folder/dataset/{store_dict_name}.json.gz')
    with gzip.open(full_store_dict_name, "rt") as f:
        store_dict = json.load(f)
    #obtain useful information
    known_names=store_dict['known_names']
    known_dtype=store_dict['known_dtype']
    known_shapes=store_dict['known_shapes']
    return full_store_dict_name, store_dict, known_names, known_dtype, known_shapes

def obtain_transformed_IR(performance_dataset_path, log_name, collected_idx, model_index, model_names):
    with gzip.open(performance_dataset_path, "rt") as f:
        performance_data = json.load(f)
    original_IR=performance_data['original_IR']
    logging.basicConfig(filename=log_name, level=logging.INFO)
    logging.info(f"model {collected_idx}/{model_index}/{len(model_names)} on strategy {strategy_index}")
    return original_IR, performance_data

def construct_single_label_sentence(full_store_dict_name, strategy_index, level, model_index, model_name, original_IR, known_info, performance_data, multiple_labels, model_dataset, dataset_entry_idx, log_name,strategy_total_num, strategy_contribution, strategy_single_label):
    dataset_sentence={'prompt':'', 'label':'', 'info': full_store_dict_name}
    prompt=f'Give the following IR of {model_name}: \'{original_IR}\', where the known variables are {known_info}. Do not change the names, shapes or dtypes of these known variables in the IR. \nPlease give me a **numerically equivalent transformed** IR that produces exactly the same outputs for any floating-point inputs (bitwise identical), and also provide applied strategy in this transformed IR. \n Return the answer **only** as a valid JSON object with the following keys:\'transformed_IR\', \'applied_strategies\'.'
    label={'transformed_IR':performance_data['transformed_IR'], 'applied_strategy':performance_data['strategy_name']}
    multiple_labels.append(label)
    dataset_sentence['prompt']=prompt
    dataset_sentence['label']=json.dumps(label)
    model_dataset.append(dataset_sentence)
    dataset_entry_idx+=1
    strategy_total_num[strategy_index]+=1
    if level+'-'+str(model_index) not in strategy_contribution[strategy_index].keys():
        strategy_contribution[strategy_index][level+'-'+str(model_index)]=1
    else:
        strategy_contribution[strategy_index][level+'-'+str(model_index)]+=1
    strategy_single_label[strategy_index].append(dataset_sentence)
    logging.basicConfig(filename=log_name, level=logging.INFO)
    logging.info(f"idx:{dataset_entry_idx}, single, data:{dataset_sentence}\n strategy_total_num:{strategy_total_num}\n strategy_contribution:{strategy_contribution}")
    # print(f"len:{dataset_entry_idx}, single, data:{dataset_sentence}\n strategy_total_num:{strategy_total_num}\n strategy_contribution:{strategy_contribution}")
    return dataset_entry_idx, model_dataset, multiple_labels, strategy_single_label

def filter_strategy_single_label(strategy_single_label, log_name):
    logging.basicConfig(filename=log_name, level=logging.INFO)
    logging.info(f"filtering")
    single_dataset=[]
    left_dataset=[]
    for strategy_index in range(44):
        all_prompt=list(set([item['prompt'] for item in strategy_single_label[strategy_index]]))
        candidate_dataset=strategy_single_label[strategy_index]
        random.shuffle(all_prompt)
        random.shuffle(candidate_dataset)
        selected_dataset=[item for prompt in all_prompt for item in candidate_dataset if item['prompt']==prompt]
        selected_dataset=selected_dataset[:min(250, len(strategy_single_label[strategy_index]))]
        # selected_dataset=random.sample(strategy_single_label[strategy_index], min(200, len(strategy_single_label[strategy_index])))
        selected_dataset_prompts=[item['prompt'] for item in selected_dataset]
        logging.basicConfig(filename=log_name, level=logging.INFO)
        logging.info(f"all_prompt:{len(all_prompt)},selected_dataset_prompts:{len(selected_dataset_prompts)},strategy_index:{strategy_index}")
        left_dataset.extend([{'prompt':item['prompt'],'info':item['info']} for item in strategy_single_label[strategy_index] if item['prompt'] not in selected_dataset_prompts])
        single_dataset.extend(selected_dataset)
    logging.basicConfig(filename=log_name, level=logging.INFO)
    logging.info(f"filtered single dataset length:{len(single_dataset)}")
    return single_dataset, left_dataset

def construct_eval_dataset(len_training_dataset, left_dataset, multi_prompt, log_name):
    num_eval_dataset=int(len_training_dataset*0.2)
    candidate_list=[dict(t) for t in {tuple(sorted(d.items())) for d in multi_prompt+left_dataset}]
    eval_dataset=random.sample(candidate_list, min(num_eval_dataset, len(candidate_list)))
    logging.basicConfig(filename=log_name, level=logging.INFO)
    logging.info(f"constructed eval dataset length:{len(candidate_list)}")
    return eval_dataset

if __name__ == '__main__':
    log_name=f'handle_dataset.log'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset=[]
    all_multi_dataset=[]
    entry_idx=0
    dataset_entry_idx=0
    strategy_total_num=[0]*44
    strategy_contribution=[{} for _ in range(44)]
    strategy_single_label=[[] for _ in range(44)]
    #model level
    for level in ['level1', 'level2']:
        model_json_path = os.path.join(script_dir, level+"_model_name.json")
        with open(model_json_path, "r") as f:
            model_names = json.load(f)
        #model name
        for model_index in range(len(model_names)):
            model_name = model_names[model_index]
            print("model_name:", model_name)
            model_dataset=[]
            #how many different input shapes for model names
            collected_num=obtain_collected_num(level, model_index)
            #different input shapes
            for collected_idx in range(collected_num):
                #load store_dict
                full_store_dict_name, store_dict, known_names, known_dtype, known_shapes=obtain_store_dict_and_known_shapes(model_name, collected_idx)
                multiple_labels=[]
                for strategy_index in range(44):
                    performance_dataset_path=f'../nfs_folder/performance_dataset/{model_name}_{collected_idx}_strategy_{strategy_index}.json.gz'
                    if os.path.exists(performance_dataset_path):
                        #obtain the dataset info
                        original_IR, performance_data = obtain_transformed_IR(performance_dataset_path, log_name, collected_idx, model_index, model_names)
                        print(f"model {collected_idx}/{model_index}/{len(model_names)} on strategy {strategy_index}")
                        known_info=obtain_known_info(known_names, known_dtype, known_shapes)
                        dataset_entry_idx, model_dataset, multiple_labels, strategy_single_label=construct_single_label_sentence(full_store_dict_name, strategy_index, level, model_index, model_name, original_IR, known_info, performance_data, multiple_labels, model_dataset, dataset_entry_idx, log_name, strategy_total_num, strategy_contribution, strategy_single_label)
                    else:
                        logging.basicConfig(filename=log_name, level=logging.INFO)
                        logging.info(f"no data: idx:{dataset_entry_idx}, model {collected_idx}/{model_index}/{len(model_names)} on strategy {strategy_index}")
                        print(f"no data: idx:{dataset_entry_idx}, model {collected_idx}/{model_index}/{len(model_names)} on strategy {strategy_index}")
                dataset_entry_idx, model_dataset, multi_dataset, multi_prompt=construct_multiple_lable_sentence(full_store_dict_name, model_dataset, multiple_labels, model_name, original_IR, known_info, log_name, dataset_entry_idx)
                all_multi_dataset.extend(multi_dataset)
            # dataset.append(model_dataset)
    single_dataset, left_dataset=filter_strategy_single_label(strategy_single_label, log_name)
    eval_dataset=construct_eval_dataset(len(single_dataset)+len(all_multi_dataset), left_dataset, multi_prompt, log_name)
    dataset.extend(single_dataset)
    dataset.extend(all_multi_dataset)
    # logging.basicConfig(filename=log_name, level=logging.INFO)
    # logging.info(f"dataset:{dataset}\n eval_dataset:{eval_dataset}\n left_dataset:{left_dataset}")

    save_dataset('train_dataset',dataset)
    save_dataset('eval_dataset',eval_dataset)
    # print(f'train_dataset:{dataset}\n eval_dataset:{eval_dataset}')
    # logging.basicConfig(filename=log_name, level=logging.INFO)
    # logging.info(f"strategy_total_num:{strategy_total_num}\n strategy_contribution:{strategy_contribution}")
    # print(f"strategy_total_num:{strategy_total_num}\n strategy_contribution:{strategy_contribution}")
    with open(f'../LLM/LLM_docker/strategy_contribution.json', "w") as f:
        json.dump(strategy_contribution, f)
    with open(f'../LLM/LLM_docker/strategy_total_num.json', "w") as f:
        json.dump(strategy_total_num, f)
    logging.basicConfig(filename=log_name, level=logging.INFO)
    logging.info(f"done.")
    

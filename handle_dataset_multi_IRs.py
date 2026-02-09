import json
import gzip
from TIR import *
from ops import *
import pickle
import tqdm
import ast

def obtain_basic_info(store_dict):
    known_names=store_dict['known_names']
    input_known_names=store_dict['input_known_names']
    known_dtype=[getattr(torch, item.split('.')[-1]) for item in store_dict['known_dtype']]
    known_shapes=[torch.Size(item) for item in store_dict['known_shapes']]
    return known_names, known_shapes, known_dtype, input_known_names


def process_one_answer(store_dict, transformed_IR):
    known_names, known_shapes, known_dtype, input_known_names=obtain_basic_info(store_dict)
    #start verification for transformed IR
    target = "cuda"
    is_built=False
    try:
        tir_string, is_built=build_tir_module_for_multi_IRs(transformed_IR, known_names, known_shapes, known_dtype, input_known_names, target)
    except Exception as e:
        tir_string=f"build_tir_module fails :{e}"
        # print(tir_string)
    return tir_string, is_built

def obtain_store_dict(info):
    with gzip.open(info, "rb") as f:
        store_dict = pickle.load(f)
    return store_dict

def handle_IR(IR, IR_to_TIR_dict, store_dict):
    delete_this_data=True
    if IR not in IR_to_TIR_dict.keys():
        tir_string, is_built=process_one_answer(store_dict, IR)
        if is_built:
            delete_this_data=False
            IR_to_TIR_dict[IR]=[tir_string, delete_this_data]
    else:
        tir_string, delete_this_data=IR_to_TIR_dict[IR]
    return tir_string, delete_this_data, IR_to_TIR_dict

if __name__=='__main__':
    with gzip.open(f'../nfs_folder/data_entries/train_dataset.json.gz', "rt") as f:
        train_dataset = json.load(f)
    print(len(train_dataset))
    IR_to_TIR_dict={}
    new_data_num=0
    #for idx in tqdm.tqdm(range(len(train_dataset)-2,len(train_dataset))):
    for idx in tqdm.tqdm(range(0,len(train_dataset))):
        #basic info
        data=train_dataset[idx]
        new_data=data.copy()
        prompt=data['prompt']
        original_IR=prompt[prompt.index(':')+2:prompt.index(', where the known variables')].strip("'")
        info=data['info'].replace('/root','..').replace('json.gz', 'pkl.gz')
        label=ast.literal_eval(data['label'])
        store_dict=obtain_store_dict(info)
        #original IR
        original_tir, delete_this_data, IR_to_TIR_dict=handle_IR(original_IR, IR_to_TIR_dict, store_dict)
        # print(f'tir_string:{original_tir},delete_this_data:{delete_this_data}')
        if not delete_this_data:
            new_data['original_IR']=original_IR
            new_data['original_TIR']=original_tir
            delete_this_data_label_side=True
            if isinstance(label, list):
                TIR_label=[]
                delete_list=[]
                for item in label:
                    item_idx=item['idx']
                    transformed_IR=item['transformed_IR'].strip("'")
                    strategy=item['applied_strategy']
                    transformed_tir, tmp_delete_this_data_label_side, IR_to_TIR_dict=handle_IR(transformed_IR, IR_to_TIR_dict, store_dict)
                    if not tmp_delete_this_data_label_side:
                        TIR_label.append({'idx':item_idx, 'transformed_TIR': transformed_tir, 'applied_strategy': strategy})
                        delete_list.append(False)
                    else:
                        delete_list=[True]
                        break
                if set(delete_list)=={False}:
                    delete_this_data_label_side=False
            else:
                transformed_IR=label['transformed_IR'].strip("'")
                strategy=label['applied_strategy']
                transformed_tir, tmp_delete_this_data_label_side, IR_to_TIR_dict=handle_IR(transformed_IR, IR_to_TIR_dict, store_dict)
                if not tmp_delete_this_data_label_side:
                    TIR_label={'transformed_TIR': transformed_tir, 'applied_strategy': strategy}
                    delete_this_data_label_side=False
            if not delete_this_data_label_side:
                new_data['TIR_label']=TIR_label
                new_data_num+=1
                with open("../nfs_folder/data_entries/multi_IRs_train_dataset.jsonl", "a") as f:
                    f.write(json.dumps(new_data, ensure_ascii=False) + "\n")
    print(f'new_data_num:{new_data_num}')

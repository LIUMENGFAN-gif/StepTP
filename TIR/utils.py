import re
import importlib
from ops import *
from strategies import *
import tvm
import numpy as np
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import inspect

related_info_pattern=[r'\^{[^}]+}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}', r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}\^{[^}]+}', r'\^{[^}]+}']
single_pattern=[r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}', r'\^{[^}]+}']
superscript_pattern=r'\^{[^}]+}'
dtype_mapping={'f32': 'float32', 'u8':'uint8', 'i8':'int8', 'u16': 'uint16',\
               'i16':'int16', 'i32': 'int32', 'i64': 'int64', 'str': 'string', \
                'bool': 'bool', 'f16': 'float16', 'f64': 'float64', 'u32': 'uint32', \
                'u64': 'uint64', 'c64': 'complex64', 'c128': 'complex128', \
                'bf16': 'bfloat16', 'f8e4m3fn': 'float8_e4m3fn', \
                'f8e4m3fnuz':'float8_e4m3fnuz', 'f8e5m2': 'float8_e5m2',\
                ' f825m2fnuz': 'float8_e5m2fnuz', 'u4': 'uint4', 'i4': 'int4'}
op_name=['exp', 'max', 'min', 'log', 'abs', 'sqrt', 'erf', 'if_then_else', 'ifthenelse']
basic_ops=['+', '-', '*', '/', '%', '=', '<', '>', '(', ',', ')', '//', '!', '&', '|']
cal_ops=['+', '-', '*', '/', '%', '(', ')', '//']
compare_ops=['=', '<', '>', '!']
thread_info={"b":"blockIdx.", "t":"threadIdx."}

def convert_cache(cache_name):
    if cache_name=="g":
        return "global"
    elif cache_name=="l":
        return "local"
    elif cache_name=="s":
        return "shared"
    return None

def convert_dtype(known_dtype):
    dtype_list=[]
    for dtype in known_dtype:
        new_dtype=dtype_mapping[dtype]
        dtype_list.append(new_dtype)
    return dtype_list

def convert_torch_dtype(known_dtype):
    dtype_list=[]
    for dtype in known_dtype:
        dtype_value=str(dtype).replace('torch.', '')
        dtype_list.append(dtype_value)
    return dtype_list

def find_subscripts_of_intermediate_vars(name, first_superscript, ir_split):
    subscript_list=[]
    start_idx=ir_split.index(name+first_superscript)
    start_this_var, start_subscript, end_subscript, end_this_var=True, False, False, False
    left_num, right_num = 0, 0
    subscript_string=''
    for idx in range(start_idx, len(ir_split)):
        if ir_split[start_idx:idx+1]==name+'^{':
            start_this_var, end_subscript, end_this_var=True, False, False
            left_num, right_num = 0, 0
            subscript_string=''
        if start_this_var and not end_subscript and not end_this_var:
            if ir_split[idx-1]=='}' and ir_split[idx]=='_' and not start_subscript:
                start_subscript=True
            elif ir_split[idx]=='{' and start_subscript:
                left_num += 1
            elif ir_split[idx]=='}' and start_subscript:
                right_num +=1
        if start_subscript and start_this_var and not end_subscript and not end_this_var:
            subscript_string+=ir_split[idx]
        if left_num == right_num and left_num > 0 and ir_split[idx]=='}' and start_this_var and start_subscript and not end_subscript and not end_this_var:
            end_subscript, start_subscript, start_this_var, end_this_var=True, False, False, True
            left_num, right_num = 0, 0
            subscript_list.append(subscript_string)
        if start_this_var and not start_subscript and ir_split[idx-1]=='}' and ir_split[idx]!='_':
            end_this_var=True
    return list(set(subscript_list))

def generate_subscript_details(subscript):
    subscript_details=[]
    left_num, right_num = 0, 0
    single_subscript=''
    for var_idx in range(2, len(subscript)-1):
        var=subscript[var_idx]
        single_subscript+=var
        if var=='{':
            left_num += 1
        elif var=='}':
            right_num += 1
        if var==',' and '{' not in single_subscript and single_subscript.replace(',', '')!='':#normal lower case
            subscript_details.append(single_subscript.replace(',', ''))
            single_subscript=''
        if var_idx==len(subscript)-2:#last string
            if '{' not in single_subscript and single_subscript.replace(',', '')!='':
                subscript_details.append(single_subscript.replace(',', ''))
            elif '{' in single_subscript and single_subscript!='':
                subscript_details.append(single_subscript)
            single_subscript=''
        if (left_num == right_num) and (left_num > 0):
            # if (('_' in single_subscript) or (var_idx!=len(subscript)-2 and subscript[var_idx+1]!='_' and subscript[var_idx+1]==',')) and single_subscript!='':
            if (var_idx!=len(subscript)-2 and subscript[var_idx+1]==',') and ('_' in single_subscript or subscript[var_idx+1]!='_' ) and single_subscript!='':
                subscript_details.append(single_subscript)
                single_subscript = ''
                left_num, right_num = 0, 0
            elif var_idx!=len(subscript)-2 and subscript[var_idx+1]=='_':
                left_num, right_num = 0, 0
    return subscript_details

def generate_non_var_subscript_details(non_var_subscript_details):
    non_var_list=[]
    for detail in non_var_subscript_details:
        non_var_list.extend(re.findall(r'[a-z]+', detail))
    return non_var_list

def generate_one_shape(non_var_list, non_var_subscript_details, varname, related_string):
    var=varname+'^{'
    # print(f'var:{var}, non_var_list: {non_var_list},related_string:{related_string}')
    shape_info={non_var_list[subspt_idx]:int(re.findall(rf'[LPVBU]\^{{([0-9]+)}}\_{{{non_var_list[subspt_idx]}=[0-9]+}}[^\]]*?{re.escape(var)}', related_string)[0])-1 for subspt_idx in range(len(non_var_list))}
    shape=[]
    if len(shape_info)>0:
        shape=[eval(expr, {}, shape_info)+1 for expr in non_var_subscript_details]
    return shape

def split_right_eq_TIR(right_eq):
    #output: new_var_list, split_eq, lower_case_var_list
    new_var_list=[]
    split_eq=[]
    lower_case_var_list=[]
    left_num, right_num = 0, 0
    has_var=False
    has_e=False
    single_eq=''
    for var_idx in range(len(right_eq)):
        var=right_eq[var_idx]
        if var.isupper():
            has_var=True
        if var=='{':
            left_num += 1
        elif var=='}':
            right_num += 1
        if has_var and (left_num == right_num) and (left_num > 0) and '^' in single_eq and (var_idx!=len(right_eq)-1 and right_eq[var_idx+1]=='_'):
            left_num, right_num = 0, 0
        single_eq += var
        if var=='e' and (var_idx!=len(right_eq)-1 and (right_eq[var_idx+1]=='-' or right_eq[var_idx+1]=='+')):
            has_e=True
        if has_e and ('+' in single_eq or '-' in single_eq) and (var_idx!=len(right_eq)-1 and right_eq[var_idx+1] in basic_ops):
            has_e=False
        #stop condition
        if var_idx==len(right_eq)-1:
            split_eq.append(single_eq)
            if single_eq.islower():
                lower_case_var_list.append(single_eq)
            if '^' in single_eq:
                new_var_list.append(single_eq)
        elif ((var_idx!=len(right_eq)-1 and right_eq[var_idx+1] in basic_ops and not has_e) or (var in basic_ops and not has_e) or single_eq in op_name) and not has_var and left_num==0 and right_num==0:
            if var_idx!=len(right_eq)-1 and right_eq[var_idx+1]=='*' and var=='*':
                continue
            split_eq.append(single_eq)
            if single_eq.islower():
                lower_case_var_list.append(single_eq)
            single_eq = ''
            has_var=False
        elif has_var and (left_num == right_num) and (left_num > 0):
            split_eq.append(single_eq)
            new_var_list.append(single_eq)
            single_eq = ''
            left_num, right_num = 0, 0
            has_var=False
    return new_var_list, split_eq, lower_case_var_list

def replace_item_in_list_using_dict(split_eq, shape_info):
    new_split_eq=[]
    for eq in split_eq:
        if eq in shape_info.keys():
            if isinstance(shape_info[eq], int):
                new_split_eq.append(str(shape_info[eq]))
            elif isinstance(shape_info[eq], list):
                new_split_eq.extend(shape_info[eq])
            elif isinstance(shape_info[eq], str):
                new_split_eq.append(shape_info[eq])
        else:
            new_split_eq.append(eq)
    return new_split_eq

def find_var_info_in_previous_ir_in_one_loop(previous_ir_list, var_subscript):
    #output: related_eq, new_var_list, previous_idx
    part_var_subscript=re.findall(rf'[A-Za-z]+{superscript_pattern}', var_subscript)[0]
    for previous_ir_idx in range(len(previous_ir_list)):
        previous_ir=previous_ir_list[previous_ir_idx]
        if part_var_subscript in previous_ir:
            related_eq=re.findall(rf'{re.escape(part_var_subscript)}[^;].*?=(.*?);', previous_ir)
            # print(f'part_var_subscript:{part_var_subscript}, related_eq:{related_eq}, previous_ir:{previous_ir}')
            if len(related_eq)>0:
                new_var_list, split_eq, lower_case_var_list=split_right_eq_TIR(related_eq[0])
                # print(f'related_eq:{related_eq}, new_var_list:{new_var_list}, split_eq:{split_eq}, lower_case_var_list:{lower_case_var_list}')
                previous_idx= previous_ir_idx
                if len(lower_case_var_list)>0:
                    shape_info={lower_case_var_list[subspt_idx]:int(re.findall(rf'[LPVBU]\^{{([0-9]+)}}\_{{{lower_case_var_list[subspt_idx]}=[0-9]+}}[^\]]*?{re.escape(part_var_subscript)}', previous_ir)[0])-1 for subspt_idx in range(len(lower_case_var_list))}
                    # print(f'previous_ir:{previous_ir}, shape_info: {shape_info}')
                    replaced_split_eq=replace_item_in_list_using_dict(split_eq, shape_info)
                    # print(f'replaced_split_eq: {replaced_split_eq}')
                    return replaced_split_eq, new_var_list, [previous_idx]*len(new_var_list)
                else:
                    return split_eq, new_var_list, [previous_ir_idx]*len(new_var_list)

def find_var_info_in_previous_ir(previous_ir_list, var_subscript):
    original_replaced_split_eq, original_var_list, original_previous_idx_list=find_var_info_in_previous_ir_in_one_loop(previous_ir_list, var_subscript)
    var_list=original_var_list.copy()
    previous_idx_list=original_previous_idx_list.copy()
    # print(f'previous_idx: {previous_idx_list}, original_replaced_split_eq: {original_replaced_split_eq}, original_var_list: {original_var_list}')
    while len(var_list)>0 and len(previous_idx_list)>0:
        new_var_subscript=var_list.pop(0)
        new_previous_idx=previous_idx_list.pop(0)
        # print(f'before var_list: {var_list}, previous_idx_list: {previous_idx_list}, new_var_subscript:{new_var_subscript}, new_previous_idx:{new_previous_idx}')
        new_replaced_split_eq, new_var_list, new_previous_idx_list=find_var_info_in_previous_ir_in_one_loop(previous_ir_list[:new_previous_idx+1], new_var_subscript)
        # print(f'new_previous_idx_list: {new_previous_idx_list}, new_replaced_split_eq: {new_replaced_split_eq}, new_var_list: {new_var_list}')
        shape_info={new_var_subscript:new_replaced_split_eq}
        original_replaced_split_eq=replace_item_in_list_using_dict(original_replaced_split_eq,shape_info)
        var_list= var_list + new_var_list
        previous_idx_list = previous_idx_list+new_previous_idx_list
        # print(f'after var_list: {var_list}, previous_idx_list: {previous_idx_list}, original_replaced_split_eq:{original_replaced_split_eq}')
    shape=eval(''.join(original_replaced_split_eq))+1
    # print(f'final shape:{shape}')
    return shape

def convert_shape(subscript_list, ir_split, name, previous_ir):
    len_shape=-1
    shape_list=[]
    correct=True
    for subscript in subscript_list:
        subscript_details= generate_subscript_details(subscript)
        # print(f'subscript_details: {subscript_details}')
        non_var_idx_list=[idx for idx in range(len(subscript_details)) if '^' not in subscript_details[idx]]
        var_idx_list=[idx for idx in range(len(subscript_details)) if '^' in subscript_details[idx]]
        final_shape_list=[-1]*len(subscript_details)
        if len(non_var_idx_list)>0:
            non_var_subscript_details=[subscript_details[idx] for idx in non_var_idx_list]
            non_var_list=generate_non_var_subscript_details(non_var_subscript_details)
            non_var_shape_list=generate_one_shape(non_var_list, non_var_subscript_details, name, ir_split)
            # print(f'non_var_shape: {non_var_shape_list}')
            for non_var_idx in range(len(non_var_idx_list)):
                final_shape_list[non_var_idx_list[non_var_idx]]=non_var_shape_list[non_var_idx]
        if len(var_idx_list)>0:
            var_subscript_details=[subscript_details[idx] for idx in var_idx_list]
            for var_subscript_idx in range(len(var_subscript_details)):
                var_subscript=var_subscript_details[var_subscript_idx]
                var_shape=find_var_info_in_previous_ir(previous_ir, var_subscript)
                final_shape_list[var_idx_list[var_subscript_idx]]=var_shape
        if len_shape==-1:
            len_shape=len(final_shape_list)
            shape_list=final_shape_list.copy()
        elif len(final_shape_list)!=len_shape:
            correct=False
        elif len(final_shape_list)==len_shape:
            if all(a >= b for a, b in zip(final_shape_list, shape_list)):
                shape_list=final_shape_list.copy()
            else:
                shape_list=[a+b for a, b in zip(shape_list, final_shape_list)]
    return shape_list, correct

def split_ir(ir):
    ir_split=[]
    left_num, right_num = 0, 0
    single_split=''
    for var in ir:
        if var=='[':
            left_num += 1
        elif var==']':
            right_num += 1
        single_split += var
        if left_num == right_num and left_num > 0 and var==';':
            ir_split.append(single_split)
            single_split = ''
            left_num, right_num = 0, 0
    return ir_split

def check_intermediate_vars(ir_split, inter_names):
    inter_names_in_this_split=[]
    for inter_name in inter_names:
        if inter_name+'^{' in ir_split:
            inter_names_in_this_split.append(inter_name)
    return inter_names_in_this_split

def add_inter_allocation(inter_dict, tab_num, inter_names_in_this_split):
    inter_string_list=[]
    inter_string_name=[]
    inter_string_shape=[]
    for name in inter_names_in_this_split:
        if name in inter_dict.keys() and inter_dict[name]['update']:
            shape, dtype, cache=inter_dict[name]['shape'], inter_dict[name]['dtype'], inter_dict[name]['cache_location']
            shape_value=str(tuple(shape))
            inter_string_list.append("\t"*tab_num+name+" = T.alloc_buffer("+shape_value+", dtype=\""+dtype+"\", scope=\""+cache+"\")\n")
            inter_string_name.append(name)
            inter_string_shape.append(shape_value)
            inter_dict[name]['update']=False
    return inter_string_list, inter_string_name, inter_string_shape

# def handle_intermediate_vars(ir_split, inter_names, inter_dict, previous_ir, tab_num):
#     inter_names_in_this_split=check_intermediate_vars(ir_split, inter_names)
#     # print(f'ir_split:{ir_split},inter_names_in_this_split:{inter_names_in_this_split}')
#     shape_correct=True
#     for name in inter_names_in_this_split:
#         superscript_list= list(set(re.findall(rf'{name}({superscript_pattern})', ir_split)))
#         subscript_list= find_subscripts_of_intermediate_vars(name, superscript_list[0], ir_split)
#         # print(f'name: {name}, superscript_list: {superscript_list}, subscript_list: {subscript_list}')
#         shape_list, shape_correct=convert_shape(subscript_list, ir_split, name, previous_ir)  
#         # print(f'shape_list: {shape_list}')
#         dtype_and_cache_index=superscript_list[0].replace('{','').replace('}','').replace('^','').split(',')
#         dtype=convert_dtype([dtype_and_cache_index[0]])[0]
#         cache=convert_cache(dtype_and_cache_index[1])
#         # print(f'dtype: {dtype}, cache: {cache}')
#         new_key_dict={'shape': shape_list, 'dtype':dtype, 'cache_location': cache, 'update': True}
#         if name not in inter_dict.keys():
#             inter_dict[name]=new_key_dict
#         else:
#             key_dict=inter_dict[name]
#             if shape_list!=key_dict['shape'] or dtype!=key_dict['dtype'] or cache!=key_dict['cache_location']:
#                 inter_dict[name]=new_key_dict
#     inter_string=add_inter_allocation(inter_dict, tab_num, inter_names_in_this_split)
#     return inter_dict, inter_string, shape_correct

def update_ir_split(temp_ir_split, full_name):
    name_start_idx=temp_ir_split.index(full_name)
    name_end_idx=name_start_idx+len(full_name)
    this_eq_end_idx= name_end_idx+temp_ir_split[name_end_idx:].index(';')
    if temp_ir_split[this_eq_end_idx+1]==']':
        this_eq_end_idx+=1
    if temp_ir_split[this_eq_end_idx+1]==';':
        if '[' in temp_ir_split[:name_start_idx-1]:
            reverse_first_part=temp_ir_split[:name_start_idx-1][::-1]
            index_bracket=len(temp_ir_split[:name_start_idx-1])-1-reverse_first_part.index('[')
            temp_ir_split=temp_ir_split[:index_bracket]+temp_ir_split[this_eq_end_idx+1:]
        else:
            temp_ir_split=temp_ir_split[:name_start_idx]+temp_ir_split[this_eq_end_idx+1:]
    else:
        temp_ir_split=temp_ir_split[:name_start_idx]+temp_ir_split[this_eq_end_idx+1:]
    return temp_ir_split

def handle_intermediate_vars(ir_split, inter_names, inter_dict, previous_ir, tab_num):
    #delete all loops
    no_loop_ir_split=re.sub(r'[LPVUB]\^{[\d]+}\_{.*?=[\d]+}','',ir_split)
    eqs=no_loop_ir_split.split(';')
    output_list=[item.split('=')[0] for item in eqs if '^' in item.split('=')[0]]
    inter_names_in_this_split=[]
    for output_item in output_list:
        inter_names_in_this_split.extend(check_intermediate_vars(output_item, inter_names))
    # print(f'output:{output_list}, no_loop_ir_split:{no_loop_ir_split}, inter_names_in_this_split:{inter_names_in_this_split}')
    # print(f'ir_split:{ir_split},inter_names_in_this_split:{inter_names_in_this_split}')
    shape_correct=True
    temp_ir_split=ir_split
    for name in inter_names_in_this_split:
        superscript_list= list(set(re.findall(rf'{name}({superscript_pattern})', temp_ir_split)))
        subscript_list= find_subscripts_of_intermediate_vars(name, superscript_list[0], temp_ir_split)
        # print(f'name: {name}, superscript_list: {superscript_list}, subscript_list: {subscript_list},temp_ir_split:{temp_ir_split}')
        shape_list, shape_correct=convert_shape(subscript_list, temp_ir_split, name, previous_ir)
        full_name=name+superscript_list[0]+subscript_list[0] if len(subscript_list)>0 else name+superscript_list[0]
        temp_ir_split=update_ir_split(temp_ir_split, full_name)
        # print(f'shape_list: {shape_list}')
        dtype_and_cache_index=superscript_list[0].replace('{','').replace('}','').replace('^','').split(',')
        dtype=convert_dtype([dtype_and_cache_index[0]])[0]
        cache=convert_cache(dtype_and_cache_index[1])
        # print(f'dtype: {dtype}, cache: {cache}')
        new_key_dict={'shape': shape_list, 'dtype':dtype, 'cache_location': cache, 'update': True}
        if name not in inter_dict.keys():
            inter_dict[name]=new_key_dict
        else:
            key_dict=inter_dict[name]
            if shape_list!=key_dict['shape'] or dtype!=key_dict['dtype'] or cache!=key_dict['cache_location']:
                inter_dict[name]=new_key_dict
    inter_string_list, inter_string_name, inter_string_shape=add_inter_allocation(inter_dict, tab_num, inter_names_in_this_split)
    return inter_dict, inter_string_list, inter_string_name, inter_string_shape, shape_correct

def check_repeated_inter(body_string, inter_name, inter_idx, inter_shape):
    if len(set(inter_name))< len(inter_name):
        # print(f'before body string:{body_string}')
        for single_name in set(inter_name):
            if inter_name.count(single_name)>1:
                idx_list=[idx for idx, name in enumerate(inter_name) if name==single_name]
                shape_list=[eval(inter_shape[idx]) for idx in idx_list]
                # print(f'shape_list:{shape_list}')
                maintain_idx=idx_list[shape_list.index(max(shape_list))]
                # print(f'inter_idx:{inter_idx},maintain_idx:{maintain_idx}, idx_list[0]:{idx_list[0]}\nbody_string:{body_string}')
                # print(f'inter_shape[maintain_idx]:{inter_shape[maintain_idx]}')
                if maintain_idx!=idx_list[0]:
                    body_string[inter_idx[idx_list[0]]]=body_string[inter_idx[maintain_idx]]
                    inter_shape[idx_list[0]]=inter_shape[maintain_idx]
                    maintain_idx=idx_list[0]
                # print(f'inter_shape[idx_list[0]]:{inter_shape[idx_list[0]]}')
                remove_idx_list=[idx for idx in idx_list if idx!=maintain_idx]
                for remove_idx in remove_idx_list:
                    body_string.pop(inter_idx[remove_idx])
                    inter_name.pop(remove_idx)
                    inter_idx.pop(remove_idx)
                    inter_shape.pop(remove_idx)
        # print(f'after body string:{body_string}')
    return body_string, inter_name, inter_idx, inter_shape

def split_loops_into_value_and_index(loops_list):
    values_list = []
    keys_list = []
    loop_type_list=[]
    for loops in loops_list:
        values_str = re.findall(r'\^{(\d+)}', loops)
        values=[int(value) for value in values_str]
        keys = re.findall(r'_\{(\w+)=', loops)
        loop_type = re.findall(r'([A-Z])\^', loops)
        values_list.append(values)
        keys_list.append(keys)
        loop_type_list.append(loop_type)
    return values_list, keys_list, loop_type_list

def distiguish_basic_loop_type(loop_type_list):
    basic_loop_type_list=[]
    num_L=0
    for loop_type_idx in range(len(loop_type_list)):
        loop_type=loop_type_list[loop_type_idx]
        if loop_type!='L' and num_L>0:
            basic_loop_type_list.append(['L',num_L])
            basic_loop_type_list.append(loop_type)
            num_L=0
        elif loop_type!='L' and num_L==0:
            basic_loop_type_list.append(loop_type)
        elif loop_type=='L':
            num_L+=1
            if loop_type_idx==len(loop_type_list)-1:
                basic_loop_type_list.append(['L',num_L])
    return basic_loop_type_list

def generate_loop_string(values_list, keys_list, basic_loop_type_list, tab_num):
    loop_string=""
    idx=0
    for loop_type in basic_loop_type_list:
        loop_string+="\t"*tab_num+"for "
        if isinstance(loop_type, list) and loop_type[0]=='L':
            _, loop_num=loop_type
            normal_key=[keys_list[normal_loop_idx] for normal_loop_idx in range(idx, idx+loop_num)]
            normal_value=[values_list[normal_loop_idx] for normal_loop_idx in range(idx, idx+loop_num)]
            if len(normal_key)>1:
                loop_string+=",".join(normal_key)+" in T.grid("
                loop_string+=",".join([str(value) for value in normal_value])+"):\n"
            else:
                loop_string+=normal_key[0]+" in range("+str(normal_value[0])+"):\n"
            idx+=loop_num
        elif isinstance(loop_type, str):
            key=keys_list[idx]
            value=values_list[idx]
            if loop_type=='P':
                loop_string+=key+" in T.parallel("+str(value)+"):\n"
            elif loop_type=='V':
                loop_string+=key+" in T.vectorized("+str(value)+"):\n"
            elif loop_type=='U':
                loop_string+=key+" in T.unroll("+str(value)+"):\n"
            elif loop_type[0]=='B':
                bind_key=thread_info[key[0]]+key[1]
                loop_string+=key+" in T.thread_binding("+str(value)+", thread=\""+bind_key+"\"):\n"
            idx+=1
        tab_num+=1
    return loop_string, tab_num

def get_loops(ir, tab_num):
    loop_match = re.findall(rf'([^\[].*?)\[', ir)[0]
    ir_split = ir[len(loop_match):]
    values_list, keys_list, loop_type_list = split_loops_into_value_and_index([loop_match])
    # print(f'loop_match: {loop_match}\nvalues_list: {values_list}, keys_list: {keys_list}, loop_type_list: {loop_type_list}')
    basic_loop_type_list=distiguish_basic_loop_type(loop_type_list[0])
    # print(f'basic_loop_type_list:{basic_loop_type_list}')
    loop_string, tab_num=generate_loop_string(values_list[0], keys_list[0], basic_loop_type_list, tab_num)
    return ir_split, loop_string, tab_num, loop_match

def record_output(simplified_output, inter_dict, output_dict, known_names, known_shapes, known_dtype):
    if simplified_output in inter_dict.keys():
        output_dict[simplified_output] = inter_dict[simplified_output]
    elif simplified_output in known_names:
        idx=known_names.index(simplified_output)
        output_dict[simplified_output]={'shape': known_shapes[idx], 'dtype': known_dtype[idx], 'cache_location': 'global', 'update': False}
    return output_dict

def judge_output_initialized(simplified_output, inter_dict, output_dict, input_known_names):
    output_initialized=False
    if simplified_output in input_known_names:
        output_initialized=True
    elif simplified_output in output_dict.keys():
        if simplified_output not in inter_dict.keys():
            output_initialized=True
        elif output_dict[simplified_output]==inter_dict[simplified_output]:
            output_initialized=True
    return output_initialized

# def get_compute(ir_split, inter_dict, output_dict, known_names, known_shapes, known_dtype, input_known_names):
#     #four cases:
#     #1. output is not in the right side
#     #2. output is in the right side, but not related to previous eq:
#     #2.1 reduction: with T.init():
#     #2.2 prefix: output initialization is given before the loop.
#     #3. output is in the right side, and related to previous eq.
#     #3.1 reduction
#     #3.2 prefix
#     #4. align case.
#     end_idx=ir_split.index(';')
#     compute_ir=ir_split[:end_idx]
#     new_ir_split=ir_split[end_idx+1:]
#     split_compute_ir=compute_ir.split('=')
#     output=split_compute_ir[0]
#     right_eq='='.join(split_compute_ir[1:])
#     if right_eq[:6]=='align(':
#         print(f'output: {output}, right_eq: {right_eq}')
#     else:
#         simplified_output=re.findall(r'([A-Za-z]+)\^{', output)[0]
#         #split the right eq
#         right_input_list, split_eq, _=split_right_eq(right_eq)
#         simplified_right_input_list=[re.findall(r'([A-Za-z]+)\^{', item)[0] for item in right_input_list]
#         simplified_input_list=[item for item in simplified_right_input_list if item!=simplified_output]
#         print(f'output: {output}, {simplified_output}, simplified_input_list: {simplified_input_list}, split_eq: {split_eq}')
#         if len(simplified_input_list)<len(simplified_right_input_list):
#             #case 2, 3
#             # output_initialized=judge_output_initialized(simplified_output, inter_dict, output_dict, input_known_names)
#             # print(f'output_initialized: {output_initialized}')
#             if output in right_input_list and simplified_output in simplified_right_input_list:
#                 #reduction
#                 pass
#             elif output not in right_input_list and simplified_output in simplified_right_input_list:
#                 #prefix
#                 pass
#         else: #case 1
#             pass
#         #record in output_dict
#         output_dict=record_output(simplified_output, inter_dict, output_dict, known_names, known_shapes, known_dtype)
#         print(f'output_dict:{output_dict}')
#     return new_ir_split

def generate_body_inputoutput_subscript_info(simplified_output,output, simplified_right_input_list, split_eq, case1):
    reads_info, writes_info=[], []
    #writes_info:
    output_superscript_list= list(set(re.findall(rf'{simplified_output}({superscript_pattern})', output)))
    output_subscript_list= find_subscripts_of_intermediate_vars(simplified_output, output_superscript_list[0], output)
    output_subscript_details= generate_subscript_details(output_subscript_list[0]) if len(output_subscript_list)>0 else []
    writes_info.append({simplified_output:output_subscript_details})
    #reads_info
    for right_item in split_eq:
        if '^{' in right_item:
            item_dict=[]
            for right_input in simplified_right_input_list:
                if right_input+'^' in right_item and (right_input!=simplified_output or (right_input==simplified_output and not case1)):
                    input_superscript_list= list(set(re.findall(rf'{right_input}({superscript_pattern})', right_item)))
                    input_subscript_list= find_subscripts_of_intermediate_vars(right_input, input_superscript_list[0], right_item)
                    input_subscript_details= generate_subscript_details(input_subscript_list[0]) if len(input_subscript_list)>0 else []
                    item_dict.append({right_input:input_subscript_details})
            reads_info.append(item_dict)
        else:
            reads_info.append(None)
    return reads_info, writes_info

def find_simplified_output_eq(previous_ir, simplified_output):
    related_eq_list=[]
    for idx in range(len(previous_ir)-1, -1, -1):
        previous_ir_item=previous_ir[idx]
        if simplified_output in previous_ir_item:
            related_eq=re.findall(rf'({re.escape(simplified_output)}[^;].*?=[^=].*?);', previous_ir_item)
            related_eq_list.extend(related_eq)
    if len(related_eq_list)>0:
        return related_eq_list, True
    return None, False

def replace_var_part_by_previous_eq(previous_ir, related_eq_list, split_sub_item):
    for replace_eq in related_eq_list:
        replace_left_part=replace_eq.split('=')[0]
        replace_right_part='='.join(replace_eq.split('=')[1:])
        replace_term=''
        if replace_left_part==split_sub_item:
            replace_term=replace_right_part
            return replace_term, True
        else:
            _, left_var_subscript_details=generate_var_subscript(replace_left_part)
            _, split_var_subscript_details=generate_var_subscript(split_sub_item)
            # print(f'left:{left_var_subscript_details}, split:{split_var_subscript_details}')
            left_original_subscript=obtain_original_subscript(previous_ir, left_var_subscript_details)
            split_original_subscript=obtain_original_subscript(previous_ir, split_var_subscript_details)
            # print(f'original: left: {left_original_subscript}, split: {split_original_subscript}')
            if ','.join(left_original_subscript)==','.join(split_original_subscript):
                replace_term=replace_right_part
                return replace_term, True
    return '', False
    

def obtain_original_subscript(previous_ir, var_subscript_details):
    original_subscript=[]
    for sub_idx in range(len(var_subscript_details)):
        sub=var_subscript_details[sub_idx].strip()
        has_replace=True
        if '^' in sub:
            while '^' in sub and has_replace:
                _,split_sub,_=split_right_eq_TIR(sub)
                for split_sub_idx in range(len(split_sub)):
                    split_sub_item=split_sub[split_sub_idx]
                    if '^' in split_sub_item:
                        simplified_split_sub_item=split_sub_item[:split_sub_item.index('^')]
                        related_eq_list, has_replace=find_simplified_output_eq(previous_ir, simplified_split_sub_item)
                        # print(f'sub:{sub}, related_eq_list: {related_eq_list}')
                        if has_replace:
                            replace_term, has_replace=replace_var_part_by_previous_eq(previous_ir, related_eq_list, split_sub_item)
                            # print(f'replace_term:{replace_term}')
                            if has_replace:
                                split_sub[split_sub_idx]=replace_term
                sub=''.join(split_sub)
        original_subscript.append(sub)
    return original_subscript

def generate_original_reduction_or_prefix_subscript(previous_ir, output_item):
    # print(f'output_item:{output_item}')
    var, var_subscript_details=generate_var_subscript(output_item)
    # print(f'in check: var: {var}, var_subscript_details: {var_subscript_details}')
    original_reduction_or_prefix_subscript=obtain_original_subscript(previous_ir, var_subscript_details)
    # print(f'original_reduction_or_prefix_subscript: {original_reduction_or_prefix_subscript}')
    return original_reduction_or_prefix_subscript

def check_if_prefix(left_output_subscript, right_output_subscript):
    prefix_index=[]
    for sub_idx in range(len(left_output_subscript)):
        left_sub=left_output_subscript[sub_idx]
        right_sub=right_output_subscript[sub_idx]
        if left_sub!=right_sub:
            if re.sub(rf'{re.escape(left_sub)}-[0-9]+','',right_sub)=='':
                prefix_num_at_idx=int(re.findall(rf'{re.escape(left_sub)}-([0-9]+)',right_sub)[0])-1
                if prefix_num_at_idx>=0:
                    prefix_index.append([sub_idx, prefix_num_at_idx])
                else:
                    # print(f'prefix_num_at_idx:{prefix_num_at_idx}')
                    return False, prefix_index
            else:
                # print(f'left_sub:{left_sub}, right_sub:{right_sub}')
                return False, prefix_index
    if len(prefix_index)==0:
        # print(f'prefix_index:{prefix_index}')
        return False, prefix_index
    return True, prefix_index

def check_sum_prod_max_min(split_eq, output_at_right):
    has_sum_prod_max_min=True
    if split_eq[0]==output_at_right and (split_eq[1]=='+' or split_eq[1]=='-'):
        init_value=str(0)
        output_location=0
    elif split_eq[0]==output_at_right and split_eq[1]=='*':
        init_value=str(1)
        output_location=0
    elif split_eq[-1]==output_at_right and (split_eq[-2]=='/' or split_eq[-2]=='//' or split_eq[-2]=='%'):
        init_value=str(1)
        output_location=-1
    elif split_eq[0]=='max' and split_eq[1]=='(' and split_eq[2]==output_at_right:
        init_value='\"-inf\"'
        output_location=2
    elif split_eq[0]=='min' and split_eq[1]=='(' and split_eq[2]==output_at_right:
        init_value='\"inf\"'
        output_location=2
    else:
        has_sum_prod_max_min=False
        init_value=None
        output_location=None
    return init_value, has_sum_prod_max_min, output_location

def check_if_reduction_or_prefix(output_initialized, previous_ir, simplified_output,output,right_input_list, split_eq):
    is_reduction, is_prefix=False, False
    init_value, output_location=None, None
    prefix_index=[]
    if not output_initialized:
        #output at left side
        left_output_subscript=generate_original_reduction_or_prefix_subscript(previous_ir, output)
        #output at right side
        output_at_right=[item for item in right_input_list if simplified_output+'^' in item]
        if len(output_at_right)>0:
            right_output_subscript=generate_original_reduction_or_prefix_subscript(previous_ir, output_at_right[0])
            if ','.join(left_output_subscript)==','.join(right_output_subscript):
                is_reduction=True
            else:
                is_prefix,prefix_index=check_if_prefix(left_output_subscript, right_output_subscript)
            init_value, has_sum_prod_max_min, output_location=check_sum_prod_max_min(split_eq, output_at_right[0])
            if not has_sum_prod_max_min:
                is_reduction, is_prefix=False, False
    # print(f'has_sum_prod_max_min:{has_sum_prod_max_min}, is_reduction:{is_reduction}, is_prefix:{is_prefix}, prefix_index:{prefix_index}, init_value:{init_value}')
    return [is_reduction, is_prefix, init_value, output_location, prefix_index]

def sub_distinguish_subscript_for_writing_and_reading(subscript, key, key_dict, index_comb_dict, index_list):
    if key not in key_dict.keys():
        key_dict[key]=[[] for _ in range(len(subscript))]
    for sub_idx in range(len(subscript)):
        sub=subscript[sub_idx]
        if '^' in sub:
            key_dict[key][sub_idx]=True
        else:
            if key_dict[key][sub_idx]!=True and sub not in key_dict[key][sub_idx]:
                key_dict[key][sub_idx].append(sub)
            single_sub=re.findall(r'[a-z]+', sub)
            if len(single_sub)>1 or (len(single_sub)==1 and len(single_sub[0])<len(sub)):
                if sub in index_comb_dict.keys():
                    index_comb_dict[sub]+=1
                else:
                    index_comb_dict[sub]=1
            index_list.extend(single_sub)
    return key_dict, index_comb_dict, index_list

def distinguish_subscript_for_writing_and_reading(reads_info, writes_info):
    reads_writes_index=[]
    reads_input_dict, writes_output_dict, index_comb_dict={}, {}, {}
    #writes_info
    write_keys=writes_info[0].keys()
    for write_key in write_keys:
        write_subscript=writes_info[0][write_key]
        writes_output_dict, index_comb_dict, reads_writes_index=sub_distinguish_subscript_for_writing_and_reading(write_subscript, write_key, writes_output_dict, index_comb_dict, reads_writes_index)
        # print(f'writes_output_dict: {writes_output_dict}, reads_writes_index: {reads_writes_index}, index_comb_dict: {index_comb_dict}')
    #reads_info
    for single_read_info in reads_info:
        if single_read_info is not None:
            for read_item in single_read_info:
                read_keys=read_item.keys()
                for read_key in read_keys:
                    read_subscript=read_item[read_key]
                    reads_input_dict, index_comb_dict, reads_writes_index=sub_distinguish_subscript_for_writing_and_reading(read_subscript, read_key, reads_input_dict, index_comb_dict, reads_writes_index)
    return list(set(reads_writes_index)), reads_input_dict, writes_output_dict, index_comb_dict

def check_S_or_R(reads_writes_index, output):
    reads_writes_property={}
    output_index=re.findall(r'[a-z]+', re.sub(r'[A-Za-z]+\^{.*?}', '', output))
    for index in reads_writes_index:
        if index in output_index:
            reads_writes_property[index]='S'
        else:
            reads_writes_property[index]='R'
    return reads_writes_property

# def check_spatial_reduce_remap(reads_input_dict, writes_output_dict, index_comb_dict, reads_writes_property):
#     spatial_dict, reduce_dict, comb_replace_dict={}, {}, {}
#     non_remap_list=[]
#     read_subs=[sub for value in reads_input_dict.values() for sublist in value if sublist!=True for sub in sublist]
#     write_subs=[sub for value in writes_output_dict.values() for sublist in value if sublist!=True for sub in sublist]
#     for index_comb in index_comb_dict.keys():
#         if index_comb_dict[index_comb]>1:
#             all_single_index=re.findall(r'[a-z]+', index_comb)
#             is_reduce, is_spatial=False, False
#             for single_index in all_single_index:
#                 if single_index not in read_subs+write_subs:
#                     if reads_writes_property[single_index]=='R':
#                         is_reduce, is_spatial=True, False
#                     elif reads_writes_property[single_index]=='S':
#                         is_reduce, is_spatial=False, True
#             if is_reduce:
#                 reduce_dict[index_comb]=['v_'+'_'.join(all_single_index)]
#                 comb_replace_dict[index_comb]='v_'+'_'.join(all_single_index)
#                 for single_index in all_single_index:
#                     if single_index not in read_subs+write_subs:
#                         non_remap_list.append(single_index)
#             elif is_spatial:
#                 spatial_dict[index_comb]=['v_'+'_'.join(all_single_index)]
#                 comb_replace_dict[index_comb]='v_'+'_'.join(all_single_index)
#                 for single_index in all_single_index:
#                     if single_index not in read_subs+write_subs:
#                         non_remap_list.append(single_index)
#             elif not is_reduce and not is_spatial:
#                 _, split_comb, lower_case_var_list=split_right_eq_TIR(index_comb)
#                 var_info={lower_case_var_list[subspt_idx]: 'v_'+lower_case_var_list[subspt_idx] for subspt_idx in range(len(lower_case_var_list))}
#                 replaced_split_comb=replace_item_in_list_using_dict(split_comb, var_info)
#                 comb_replace_dict[index_comb]=''.join(replaced_split_comb)
#     return spatial_dict, reduce_dict, list(set(non_remap_list)), comb_replace_dict

def check_spatial_reduce_remap(reads_input_dict, writes_output_dict, index_comb_dict, reads_writes_property, single_index_in_split_eq):
    spatial_dict, reduce_dict, comb_replace_dict={}, {}, {}
    non_remap_list=[]
    read_subs=[sub for value in reads_input_dict.values() for sublist in value if sublist!=True for sub in sublist]
    write_subs=[sub for value in writes_output_dict.values() for sublist in value if sublist!=True for sub in sublist]
    # print(f'read_subs: {read_subs}, write_subs: {write_subs}')
    for index_comb in index_comb_dict.keys():
        reduce_space_list=[]
        is_reduce, is_spatial=False, False
        all_single_index=re.findall(r'[a-z]+', index_comb)
        all_vars=re.findall(r'[a-z]+|[0-9]+', index_comb)
        for single_index in all_single_index:
            if reads_writes_property[single_index]=='R':
                reduce_space_list.append('R')
            elif reads_writes_property[single_index]=='S':
                reduce_space_list.append('S')
        # print(f'index_comb: {index_comb}, reduce_space_list: {reduce_space_list}, all_single_index: {all_single_index}')
        if index_comb_dict[index_comb]>=1 and len(set(reduce_space_list))==1:
            if reduce_space_list[0]=='R':
                is_reduce, is_spatial=True, False
            elif reduce_space_list[0]=='S':
                is_reduce, is_spatial=False, True
            new_name='v_'+'_'.join(all_vars)
            if is_reduce:
                while new_name in comb_replace_dict.values():
                    new_name=new_name+'_1'
                reduce_dict[index_comb]=[new_name]
                comb_replace_dict[index_comb]=new_name
                for single_index in all_single_index:
                    if single_index not in read_subs+write_subs and single_index not in single_index_in_split_eq:
                        non_remap_list.append(single_index)
            elif is_spatial:
                while new_name in comb_replace_dict.values():
                    new_name=new_name+'_1'
                spatial_dict[index_comb]=[new_name]
                comb_replace_dict[index_comb]=new_name
                for single_index in all_single_index:
                    if single_index not in read_subs+write_subs and single_index not in single_index_in_split_eq:
                        non_remap_list.append(single_index)
        if not is_reduce and not is_spatial:
            for single_index in all_single_index:
                if single_index in non_remap_list:
                    non_remap_list.remove(single_index)
                    single_index_in_split_eq.append(single_index)
                elif single_index not in single_index_in_split_eq:
                    single_index_in_split_eq.append(single_index)
            _, split_comb, lower_case_var_list=split_right_eq_TIR(index_comb)
            var_info={lower_case_var_list[subspt_idx]: 'v_'+lower_case_var_list[subspt_idx] for subspt_idx in range(len(lower_case_var_list)) if lower_case_var_list[subspt_idx] not in op_name}
            replaced_split_comb=replace_item_in_list_using_dict(split_comb, var_info)
            comb_replace_dict[index_comb]=''.join(replaced_split_comb)
    return spatial_dict, reduce_dict, list(set(non_remap_list)), comb_replace_dict

def look_up_var_shape(simplified_var, inter_dict, known_names, known_shapes):
    if simplified_var in inter_dict.keys():
        shape = inter_dict[simplified_var]['shape']
    elif simplified_var in known_names:
        idx=known_names.index(simplified_var)
        shape=known_shapes[idx]
    return shape

def look_up_var_dtype(simplified_var, inter_dict, known_names, known_dtype):
    if simplified_var in inter_dict.keys():
        dtype = inter_dict[simplified_var]['dtype'].replace("torch.","")
    elif simplified_var in known_names:
        idx=known_names.index(simplified_var)
        dtype=str(known_dtype[idx]).replace("torch.","")
    return dtype

def sub_generate_shape_remap_and_write_read_list(key_dict, reduce_dict, spatial_dict, comb_replace_dict, inter_dict, known_names, known_shapes):
    key_list=[]
    for key in key_dict.keys():
        subscripts= key_dict[key]
        # print(f'key: {key}\n inter_dict: {inter_dict}\n known_names: {known_names}\n known_shapes: {known_shapes}')
        shape=look_up_var_shape(key, inter_dict, known_names, known_shapes)
        write_read_sub=[[] for _ in range(len(subscripts))]
        for sub_idx in range(len(subscripts)):
            sub=subscripts[sub_idx]
            if sub==True:
                write_read_sub[sub_idx]='0:'+ str(shape[sub_idx])
            else:
                len_sub=len(sub)
                new_sub=[]
                for sub_item in sub:
                    if sub_item in comb_replace_dict.keys():
                        new_sub.append(comb_replace_dict[sub_item])
                        if sub_item in reduce_dict.keys() and len(reduce_dict[sub_item])==1:
                            reduce_dict[sub_item].append(shape[sub_idx])
                        elif sub_item in spatial_dict.keys() and len(spatial_dict[sub_item])==1:
                            spatial_dict[sub_item].append(shape[sub_idx])
                    elif sub_item.islower() and sub_item.isalpha():
                        new_sub.append('v_'+sub_item)
                    else:
                        _, split_sub_item, _=split_right_eq_TIR(sub_item)
                        new_split_sub=[]
                        for split_sub in split_sub_item:
                            if split_sub.islower() and split_sub.isalpha():
                                new_split_sub.append('v_'+split_sub)
                            else:
                                new_split_sub.append(split_sub)
                        new_sub.append(''.join(new_split_sub))
                if len_sub==1:
                    write_read_sub[sub_idx]=new_sub[0]
                elif len(new_sub)>1:
                    min_item='T.min(' + ','.join(new_sub) + ')'
                    max_item='T.max(' + ','.join(new_sub) + ')'
                    write_read_sub[sub_idx]=min_item+':'+max_item+'+1'
        if len(write_read_sub)>0:
            key_item=key+'['+','.join(write_read_sub)+']'
        else:
            key_item=key+'[()]'
        key_list.append(key_item)
    return key_list, reduce_dict, spatial_dict
    
def generate_remap_list(reads_writes_index, reads_writes_property, non_remap_list):
    index_list=[]
    reexpr_list=[]
    property_string=""
    for index in reads_writes_index:
        if index not in non_remap_list and index!='inf':
            index_list.append(index)
            reexpr_list.append('v_'+index)
            property_string+=reads_writes_property[index]
    return [reexpr_list, index_list, property_string]

def generate_shape_remap_and_write_read_list(reads_writes_index, reads_writes_property, reduce_dict, spatial_dict, non_remap_list, comb_replace_dict, reads_input_dict, writes_output_dict, inter_dict, known_names, known_shapes):
    #writes info
    writes_list, reduce_dict, spatial_dict=sub_generate_shape_remap_and_write_read_list(writes_output_dict, reduce_dict, spatial_dict, comb_replace_dict,inter_dict, known_names, known_shapes)
    #reads info
    reads_list, reduce_dict, spatial_dict=sub_generate_shape_remap_and_write_read_list(reads_input_dict, reduce_dict, spatial_dict, comb_replace_dict, inter_dict, known_names, known_shapes)
    remap_list=generate_remap_list(reads_writes_index, reads_writes_property, non_remap_list)
    return reduce_dict, spatial_dict, remap_list, writes_list, reads_list

def generate_write_read_list(single_index_in_split_eq, reads_writes_index,reads_input_dict,writes_output_dict,index_comb_dict,inter_dict,known_names,known_shapes, output):
    all_index_list=list(set(single_index_in_split_eq + reads_writes_index))
    #check_S_or_R
    reads_writes_property=check_S_or_R(all_index_list, output)
    # print(f'reads_writes_property: {reads_writes_property}')
    #check spatial, reduce, remap
    spatial_dict, reduce_dict, non_remap_list, comb_replace_dict=check_spatial_reduce_remap(reads_input_dict, writes_output_dict, index_comb_dict, reads_writes_property, single_index_in_split_eq)
    # print(f'comb_replace_dict:{comb_replace_dict}')
    # print(f'reduce_dict: {reduce_dict}, spatial_dict: {spatial_dict}, non_remap_list: {non_remap_list}, single_index_in_split_eq:{single_index_in_split_eq}')
    #shape and write_read_list
    reduce_dict, spatial_dict, remap_list, writes_list, reads_list=generate_shape_remap_and_write_read_list(all_index_list,reads_writes_property, reduce_dict, spatial_dict, non_remap_list, comb_replace_dict, reads_input_dict,writes_output_dict,inter_dict,known_names,known_shapes)
    return reduce_dict, spatial_dict, remap_list, writes_list, reads_list, comb_replace_dict

def obtain_write_read_string(reduce_dict, spatial_dict, remap_list, writes_list, reads_list, temp_tab_num):
    write_read_string=""
    #reduce_dict
    for reduce_key in reduce_dict.keys():
        reexpress, reduce_shape=reduce_dict[reduce_key]
        write_read_string+="\t"*temp_tab_num+reexpress+'=T.axis.reduce('+str(reduce_shape)+','+ reduce_key+')\n'
    #spatial_dict
    for spatial_key in spatial_dict.keys():
        reexpress, spatial_shape=spatial_dict[spatial_key]
        write_read_string+="\t"*temp_tab_num+reexpress+'=T.axis.spatial('+str(spatial_shape)+','+ spatial_key+')\n'
    #remap_list
    if len(remap_list)>0:
        reexpress_list, index_list, property_string=remap_list
        if len(reexpress_list)>0:
            write_read_string+="\t"*temp_tab_num+','.join(reexpress_list)+'=T.axis.remap(\"'+property_string+'\",['+','.join(index_list)+'])\n'
    #reads_list
    if len(reads_list)>0:
        write_read_string+="\t"*temp_tab_num+'T.reads('+','.join(reads_list)+')\n'
    else:
        write_read_string+="\t"*temp_tab_num+'T.reads()\n'
    #writes_list
    if len(writes_list)>0:
        write_read_string+="\t"*temp_tab_num+'T.writes('+','.join(writes_list)+')\n'
    return write_read_string

# def generate_index_in_split_eq(split_eq, index_comb_dict):
#     add, start=False, False
#     index_in_split_eq=[]
#     single_index_in_split_eq=[]
#     left_num, right_num=0, 0
#     single_item=''
#     for split_eq_item in split_eq:
#         split_eq_item=split_eq_item.strip()
#         if split_eq_item.islower() and split_eq_item.isalpha() and split_eq_item not in op_name:
#             add, start=True, True
#             if not add:
#                 single_item=''
#                 left_num, right_num=0, 0
#             single_index_in_split_eq.append(split_eq_item)
#         elif split_eq_item in cal_ops:
#             add=True
#         elif re.fullmatch(r'-?\d+(\.\d+)?', split_eq_item):
#             add=True
#         else:
#             add, start=False, False
#         # print(f'split_eq_item: {split_eq_item}, add: {add}, start: {start}, single_item:{single_item}')
#         if add and start:
#             single_item+=split_eq_item
#             if split_eq_item=='(':
#                 left_num += 1
#             elif split_eq_item==')':
#                 right_num += 1
#         if left_num==right_num and not add and not start and single_item!='':
#             if single_item not in single_index_in_split_eq:
#                 index_in_split_eq.append(single_item)
#                 if single_item in index_comb_dict.keys():
#                     index_comb_dict[single_item]+=1
#                 else:
#                     index_comb_dict[single_item]=1
#             single_item=''
#     single_index_in_split_eq=list(set(single_index_in_split_eq))
#     index_in_split_eq=list(set(index_in_split_eq))
#     print(f'split_eq: {split_eq}, single_index_in_split_eq:{single_index_in_split_eq}, index_in_split_eq: {index_in_split_eq}, index_comb_dict:{index_comb_dict}')

def generate_index_in_split_eq(split_eq):
    single_index_in_split_eq=[]
    for split_eq_item in split_eq:
        split_eq_item=split_eq_item.strip()
        if split_eq_item.islower() and split_eq_item.isalpha() and split_eq_item not in op_name and split_eq_item!='inf':
            single_index_in_split_eq.append(split_eq_item)
    single_index_in_split_eq=list(set(single_index_in_split_eq))
    return single_index_in_split_eq

def generate_write_read_string(split_eq, reads_info, writes_info, inter_dict, known_names, known_shapes, output, temp_tab_num):
    reads_writes_index,reads_input_dict,writes_output_dict,index_comb_dict= distinguish_subscript_for_writing_and_reading(reads_info, writes_info)
    # print(f'reads_writes_index: {reads_writes_index}, reads_input_dict: {reads_input_dict}, writes_output_dict: {writes_output_dict}, index_comb_dict: {index_comb_dict}')
    single_index_in_split_eq=generate_index_in_split_eq(split_eq)
    reduce_dict, spatial_dict, remap_list, writes_list, reads_list, comb_replace_dict=generate_write_read_list(single_index_in_split_eq, reads_writes_index,reads_input_dict,writes_output_dict,index_comb_dict,inter_dict,known_names,known_shapes,output)
    # print(f'reads_list: {reads_list}, writes_list: {writes_list}\nreduce_dict: {reduce_dict}, spatial_dict: {spatial_dict}, remap_list: {remap_list}')
    write_read_string=obtain_write_read_string(reduce_dict, spatial_dict, remap_list, writes_list, reads_list, temp_tab_num)
    return write_read_string, comb_replace_dict

def handle_compute_string(var_key, var_subscript_list, point_sub_idx_in_last_sub, comb_replace_dict):
    compute_subscript=[]
    var_list=[]
    var_subscript_details_list=[]
    var_idx_list=[]
    # print(f'var_key:{var_key}, var_subscript_list:{var_subscript_list}, comb_replace_dict:{comb_replace_dict}')
    for sub_idx in range(len(var_subscript_list)):
        sub=var_subscript_list[sub_idx].strip()
        if sub.islower() and sub.isalpha():
            compute_subscript.append('v_'+sub)
        elif sub in comb_replace_dict.keys():
            compute_subscript.append(comb_replace_dict[sub])
        elif '^' in sub:
            _, split_sub, _=split_right_eq_TIR(sub)
            # print(f'split_sub: {split_sub}')
            for split_sub_idx in range(len(split_sub)):
                split_sub_item=split_sub[split_sub_idx]
                if '^' in split_sub_item:
                    var, var_subscript_details=generate_var_subscript(split_sub_item)
                    var_list.append(var)
                    var_subscript_details_list.append(var_subscript_details)
                    var_idx_list.append([sub_idx,split_sub_idx])
                    split_sub[split_sub_idx]=split_sub_item
                elif split_sub_item in comb_replace_dict.keys():
                    split_sub[split_sub_idx]=comb_replace_dict[split_sub_item]
                elif split_sub_item.islower() and split_sub_item.isalpha():
                    split_sub[split_sub_idx]='v_'+split_sub_item
            compute_subscript.append(''.join(split_sub))
        else:
            _, split_sub, _=split_right_eq_TIR(sub)
            for split_sub_idx in range(len(split_sub)):
                split_sub_item=split_sub[split_sub_idx]
                if split_sub_item in comb_replace_dict.keys():
                    split_sub[split_sub_idx]=comb_replace_dict[split_sub_item]
                elif split_sub_item.islower() and split_sub_item.isalpha():
                    split_sub[split_sub_idx]='v_'+split_sub_item
                else:
                    split_sub[split_sub_idx]=split_sub_item
            compute_subscript.append(''.join(split_sub))
    return [var_key, compute_subscript, point_sub_idx_in_last_sub, var_list, var_subscript_details_list, var_idx_list]
    

def generate_var_subscript(sub):
    # print(f'sub: {sub}')
    var=sub[:sub.index('^')]
    var_superscript_list= list(set(re.findall(rf'{var}({superscript_pattern})', sub)))
    var_subscript_list= find_subscripts_of_intermediate_vars(var, var_superscript_list[0], sub)
    var_subscript_details= generate_subscript_details(var_subscript_list[0]) if len(var_subscript_list)>0 else []
    return var, var_subscript_details

def generate_compute_subscript(split_eq_item, comb_replace_dict):
    var, var_subscript_details=generate_var_subscript(split_eq_item)
    var_related=handle_compute_string(var, var_subscript_details, [-1,-1], comb_replace_dict)
    var_related.append(-1)
    var_related_list=[var_related]
    # print(f'var_related_list: {var_related_list}')
    while len(var_related_list[-1][3])>0:
        _, _, _,var_list, var_subscript_details_list, var_idx_list, _= var_related_list[-1]
        var_related_idx=len(var_related_list)-1
        for idx in range(len(var_list)):
            var=var_list[idx]
            var_subscript_details=var_subscript_details_list[idx]
            sub_idx_and_split_sub_idx=var_idx_list[idx]
            new_var_related=handle_compute_string(var, var_subscript_details, sub_idx_and_split_sub_idx, comb_replace_dict)
            new_var_related.append(var_related_idx)
            var_related_list.append(new_var_related)
    for var_related_idx in range(len(var_related_list)-1, 0, -1):
        var_key, compute_subscript, point_sub_idx_in_last_sub, _, _, _, point_var_related_idx=var_related_list[var_related_idx]
        this_var=var_key+'['+','.join(compute_subscript)+']'
        sub_idx,split_sub_idx=point_sub_idx_in_last_sub
        # print(f'before split:{var_related_list[point_var_related_idx][1][sub_idx]}')
        _, split_sub, _=split_right_eq_TIR(var_related_list[point_var_related_idx][1][sub_idx])
        split_sub[split_sub_idx]=this_var
        var_related_list[point_var_related_idx][1][sub_idx]=''.join(split_sub)
        # print(f'after split:{var_related_list[point_var_related_idx][1][sub_idx]}')
    if len(var_related_list[0][1])>0:
        compute_subscript=var_related_list[0][0]+'['+','.join(var_related_list[0][1])+']'
    else:
        compute_subscript=var_related_list[0][0]+'[()]'
    return compute_subscript

def generate_compute_split_eq(split_eq, comb_replace_dict, output_dtype):
    compute_split_eq=[]
    has_if_then_else=False
    has_pow=False
    pow_bracket_num=0
    # print(f'split_eq:{split_eq}')
    for split_eq_item in split_eq:
        if split_eq_item in op_name:
            compute_split_eq.append('T.'+split_eq_item)
            if split_eq_item=='if_then_else':
                has_if_then_else=True
        elif re.fullmatch(r'-?\d+(\.\d+)?', split_eq_item) or ('e' in split_eq_item and split_eq_item.islower()):
            if has_if_then_else:
                compute_split_eq.append(split_eq_item)
            elif has_pow:
                if compute_split_eq[-1]==')' and pow_bracket_num==0:
                    compute_split_eq.insert(-1,',T.'+output_dtype+'('+split_eq_item+')')
                    has_pow=False
                elif compute_split_eq[-1]=='(' and compute_split_eq[-2]!='(':
                    compute_split_eq.insert(-1,',')
                    compute_split_eq.append('T.'+output_dtype+'('+split_eq_item+')')
                else:
                    if pow_bracket_num==0:
                        compute_split_eq.append(',T.'+output_dtype+'('+split_eq_item+'))')
                        has_pow=False
                    else:
                        compute_split_eq.append('T.'+output_dtype+'('+split_eq_item+')')
            else:
                compute_split_eq.append('T.'+output_dtype+'('+split_eq_item+')')
        elif split_eq_item=='inf':
            compute_split_eq.append('T.float64(\"inf\")')
        elif split_eq_item in basic_ops:
            if split_eq_item==',':
                has_if_then_else=False
                compute_split_eq.append(split_eq_item)
            elif split_eq_item=='(' and has_pow:
                pow_bracket_num+=1
                compute_split_eq.append(split_eq_item)
            elif split_eq_item==')' and has_pow:
                pow_bracket_num-=1
                compute_split_eq.append(split_eq_item)
                if pow_bracket_num==0:
                    compute_split_eq.append(')')
                    has_pow=False
            elif has_if_then_else and split_eq_item=='&':
                compute_split_eq.append(' and ')
            elif has_if_then_else and split_eq_item=='|':
                compute_split_eq.append(' or ')
            elif has_if_then_else and (len(compute_split_eq)>=2 and compute_split_eq[-2] in compare_ops) and (split_eq_item in compare_ops):
                # print(f'compute_split_eq before1: {compute_split_eq}')
                compute_split_eq.append(' and ')
                compute_split_eq.append(compute_split_eq[-2])
                compute_split_eq.append(split_eq_item)
            else:
                compute_split_eq.append(split_eq_item)
        elif split_eq_item=='**':
            # print(f'compute_split_eq before2: {compute_split_eq}')
            if compute_split_eq[-1]==')':
                last_bracket_idx=len(compute_split_eq) - 1 - compute_split_eq[::-1].index('(')
                compute_split_eq.insert(last_bracket_idx,'T.pow')
            else:
                compute_split_eq.insert(-1,'T.pow(')
            # print(f'compute_split_eq after2: {compute_split_eq}')
            has_pow=True
        elif split_eq_item.islower() and split_eq_item.isalpha() and split_eq_item!= 'inf':
            if has_if_then_else:
                compute_split_eq.append('v_'+split_eq_item)
            else:
                compute_split_eq.append('T.Cast(\"'+output_dtype+'\",v_'+split_eq_item+')')
        elif '^' in split_eq_item:
            compute_subscript=generate_compute_subscript(split_eq_item, comb_replace_dict)
            compute_split_eq.append(compute_subscript)
        else:
            compute_split_eq.append(None)
    # print(f'compute_split_eq: {compute_split_eq}')
    return compute_split_eq

def generate_init_string(init_info, right_compute_split_eq, output_dtype, comb_replace_dict, temp_tab_num):
    is_reduction, is_prefix, init_value, output_location, prefix_index=init_info
    init_string=""
    if is_reduction or is_prefix:
        output_term=right_compute_split_eq[output_location]
        # print(f'output_term:{output_term}')
        if is_reduction:
            init_string+="\t"*temp_tab_num+'with T.init():\n'
            init_string+="\t"*(temp_tab_num+1)+output_term+'=T.'+output_dtype+'('+init_value+')\n'
        elif is_prefix:
            output_term_subscript=output_term[output_term.index('[')+1:-1]
            output_term_subscript_details=output_term_subscript.split(',')
            condition_string=""
            for prefix_idx in range(len(prefix_index)-1):
                sub_idx, prefix_num_at_idx=prefix_index[prefix_idx]
                condition_string+=output_term_subscript_details[sub_idx]+'<'+ str(prefix_num_at_idx)+' and '
            sub_idx, prefix_num_at_idx=prefix_index[-1]
            condition_string+=output_term_subscript_details[sub_idx]+'<'+ str(prefix_num_at_idx)
            init_string+="\t"*temp_tab_num+'with T.init():\n'
            init_string+="\t"*(temp_tab_num+1)+output_term+'=T.if_then_else('+condition_string+','+output_term+',T.'+output_dtype+'('+init_value+'))\n'
    return init_string

def generate_compute_string(output, split_eq, comb_replace_dict, simplified_output, inter_dict, known_names, known_dtype, output_initialized, init_info, temp_tab_num):
    compute_string=""
    #output dtype
    output_dtype=look_up_var_dtype(simplified_output, inter_dict, known_names, known_dtype)
    #right_eq
    right_compute_split_eq=generate_compute_split_eq(split_eq, comb_replace_dict, output_dtype)
    # print(f'split_eq:{split_eq}, right_compute_split_eq: {right_compute_split_eq}')
    #left_eq
    left_compute_split_eq=generate_compute_split_eq([output], comb_replace_dict, output_dtype)
    # print(f'left_compute_split_eq: {left_compute_split_eq}')
    if not output_initialized:
        init_string=generate_init_string(init_info, right_compute_split_eq, output_dtype, comb_replace_dict, temp_tab_num)
        compute_string+=init_string
    compute_eq=left_compute_split_eq[0]+'='+''.join(right_compute_split_eq)
    # print(f'compute_eq: {compute_eq}')
    compute_string+="\t"*temp_tab_num+compute_eq+'\n'
    return compute_string

# def get_compute(ir_split, inter_dict, output_dict, known_names, known_shapes, known_dtype, input_known_names, temp_tab_num):
#     end_idx=ir_split.index(';')
#     compute_ir=ir_split[:end_idx]
#     new_ir_split=ir_split[end_idx+1:]
#     split_compute_ir=compute_ir.split('=')
#     output=split_compute_ir[0]
#     right_eq='='.join(split_compute_ir[1:])
#     compute_string=""
#     simplified_output=re.findall(r'([A-Za-z]+)\^{', output)[0]
#     compute_string+="\t"*temp_tab_num+'with T.block(\"'+simplified_output+'\"):\n'
#     temp_tab_num+=1
#     #split the right eq
#     right_input_list, split_eq, _=split_right_eq(right_eq)
#     # print(f'right_input_list:{right_input_list}')
#     simplified_right_input_list=list(set([reitem for item in right_input_list for reitem in re.findall(r'([A-Za-z]+)\^{', item)]))
#     simplified_input_list=[item for item in simplified_right_input_list if item!=simplified_output]
#     # print(f'output: {output}, {simplified_output}, simplified_right_input_list: {simplified_right_input_list}, split_eq: {split_eq}')
#     if len(simplified_input_list)<len(simplified_right_input_list):
#         output_initialized=judge_output_initialized(simplified_output, inter_dict, output_dict, input_known_names)
#         # print(f'output_initialized: {output_initialized}')
#         case1=False if output_initialized else True
#     else: #case 1
#         case1=True
#     #record in output_dict
#     output_dict=record_output(simplified_output, inter_dict, output_dict, known_names, known_shapes, known_dtype)
#     # print(f'output_dict:{output_dict}')
#     reads_info, writes_info = generate_body_inputoutput_subscript_info(simplified_output,output, simplified_right_input_list, split_eq, case1)
#     # print(f'reads_info: {reads_info}, writes_info: {writes_info}')
#     write_read_string, comb_replace_dict=generate_write_read_string(reads_info, writes_info, inter_dict, known_names, known_shapes, output, temp_tab_num)
#     compute_string+=write_read_string
#     expr_string=generate_compute_string(output, split_eq, comb_replace_dict, simplified_output, inter_dict, known_names, known_dtype, temp_tab_num)
#     compute_string+=expr_string
#     return new_ir_split, compute_string

def get_compute(previous_ir, ir_split, inter_dict, output_dict, known_names, known_shapes, known_dtype, input_known_names, temp_tab_num):
    end_idx=ir_split.index(';')
    compute_ir=ir_split[:end_idx]
    new_ir_split=ir_split[end_idx+1:]
    split_compute_ir=compute_ir.split('=')
    output=split_compute_ir[0]
    # print(f'output: {output}, compute_ir: {compute_ir}')
    right_eq='='.join(split_compute_ir[1:])
    compute_string=""
    simplified_output=re.findall(r'([A-Za-z]+)\^{', output)[0]
    compute_string+="\t"*temp_tab_num+'with T.block(\"'+simplified_output+'\"):\n'
    temp_tab_num+=1
    #split the right eq
    right_input_list, split_eq, _=split_right_eq_TIR(right_eq)
    # print(f'here is the split_eq: {split_eq}')
    # print(f'right_input_list:{right_input_list}')
    # print(f'split_eq: {split_eq}')
    simplified_right_input_list=list(set([reitem for item in right_input_list for reitem in re.findall(r'([A-Za-z]+)\^{', item)]))
    simplified_input_list=[item for item in simplified_right_input_list if item!=simplified_output]
    # print(f'output: {output}, {simplified_output}, simplified_right_input_list: {simplified_right_input_list}, split_eq: {split_eq}')
    if len(simplified_input_list)<len(simplified_right_input_list):
        output_initialized=judge_output_initialized(simplified_output, inter_dict, output_dict, input_known_names)
        # print(f'output_initialized: {output_initialized}')
        case1=False if output_initialized else True
        init_info=check_if_reduction_or_prefix(output_initialized, previous_ir, simplified_output,output,right_input_list, split_eq)
    else: #case 1
        output_initialized=False
        case1=True
        init_info=[False, False, None, None, []]  # [is_reduction, is_prefix, init_value, output_location, prefix_index]
    #record in output_dict
    output_dict=record_output(simplified_output, inter_dict, output_dict, known_names, known_shapes, known_dtype)
    # print(f'output_dict:{output_dict}')
    reads_info, writes_info = generate_body_inputoutput_subscript_info(simplified_output,output, simplified_right_input_list, split_eq, case1)
    # print(f'reads_info: {reads_info}, writes_info: {writes_info}')
    write_read_string, comb_replace_dict=generate_write_read_string(split_eq, reads_info, writes_info, inter_dict, known_names, known_shapes, output, temp_tab_num)
    compute_string+=write_read_string
    expr_string=generate_compute_string(output, split_eq, comb_replace_dict, simplified_output, inter_dict, known_names, known_dtype, output_initialized, init_info, temp_tab_num)
    compute_string+=expr_string
    return new_ir_split, compute_string, output_dict

def select_shape(model_name):
    module = importlib.import_module("model_codes")
    set_default_shapes_ranges_and_dtypes = getattr(module, f"{model_name}_set_default_shapes_ranges_and_dtypes", None)
    default_shapes, ranges, dtypes=set_default_shapes_ranges_and_dtypes()
    return default_shapes, ranges, dtypes, module

def obtain_real_inputs_for_verification(module, model_name, input_shapes, device, dtype):
    get_real_inputs = getattr(module, f"{model_name}_get_real_inputs", None)
    real_inputs_on_cpu = get_real_inputs(*input_shapes, dtype=dtype)
    real_inputs=[]
    print('input item start')
    for input_item in real_inputs_on_cpu:
        print(f'input_item:{type(input_item)}')
        if isinstance(input_item, torch.Tensor):
            input_item = input_item.to(device)
        elif isinstance(input_item, float):
            input_item = torch.tensor(input_item, dtype=dtype).to(device)
        real_inputs.append(input_item)
    print('input item done')
    if not isinstance(real_inputs, tuple):
        real_inputs = tuple(real_inputs)
    return real_inputs

def generate_ctx_and_device(target):
    if '-1' in target:
        return tvm.cuda(), 'cuda'
    if 'cuda' in target:
        cuda_num_list=re.findall(r'-device=([0-9]+)',target)
        if len(cuda_num_list)>0:
            cuda_num=int(cuda_num_list[0])
        else:
            cuda_num=0
        print(f'cuda_num:{cuda_num}')
        ctx=tvm.cuda(cuda_num)
        device='cuda:'+str(cuda_num)
        print(f'ctx:{ctx}')
        return ctx, device
    else:
        return tvm.cpu(), 'cpu'
    
def generate_f_input_list(ctx, real_inputs,constant_params_value, output_shape, output_dtype):
    input_list=[]
    for input_tensor in real_inputs:
        input_list.append(tvm.nd.array(input_tensor.detach().cpu().numpy(), ctx))
    for constant_value in constant_params_value:
        if isinstance(constant_value, torch.Tensor):
            input_list.append(tvm.nd.array(constant_value.detach().cpu().numpy(), ctx))
        else:
            input_list.append(tvm.nd.array(np.array(constant_value), ctx))
    if 'float' in str(output_dtype):
        output_placeholder=torch.randn(output_shape, dtype=output_dtype)
    elif 'int' in str(output_dtype):
        output_placeholder=torch.randint(0, 10, output_shape, dtype=output_dtype)
    input_list.append(tvm.nd.array(output_placeholder.detach().cpu().numpy(), ctx))
    return input_list

def check_if_two_outputs_equal(tvm_output, torch_output, atol=1e-3, rtol=0):
    torch_output_np = torch_output.cpu().detach().numpy()
    if np.allclose(tvm_output, torch_output_np, atol=atol, rtol=rtol):
        info= f"Outputs are equal with atol={atol} and rtol={rtol}."
        return True, info
    else:
        info = f"Outputs are not equal with atol={atol} and rtol={rtol}."
        return False, info

def generate_random_shape(default_shapes, ranges, dtypes, use_defualt):
    if use_defualt:
        return default_shapes
    else:
        shape_list=[]
        for idx in range(len(ranges)):
            if len(ranges[idx])==2:
                low, high = ranges[idx]
                dtype= dtypes[idx]
                if dtype==int:
                    shape=np.random.randint(low, high, dtype=dtype)
                elif dtype==float:
                    shape=np.random.uniform(low, high)
            else:
                shape=random.choice(ranges[idx])
            shape_list.append(shape)
        return shape_list

def select_model_with_new_shapes(model_name, new_shapes, dtype):
    params_name=[]
    params_shape=[]
    params_dtype=[]
    params_value_dict={}
    module = importlib.import_module("model_codes")
    split_shapes_into_input_and_model_params_shapes = getattr(module, f"{model_name}_split_shapes_into_input_and_model_params_shapes", None)
    get_inputs = getattr(module, f"{model_name}_get_inputs")
    get_model = getattr(module, f"{model_name}_get_model")
    input_shapes, model_params_shapes = split_shapes_into_input_and_model_params_shapes(*new_shapes)
    print('getting inputs')
    inputs = get_inputs(*input_shapes, dtype=dtype)
    if inputs is not None:
        print('getting model')
        model = get_model(*model_params_shapes, dtype=dtype)
        model.eval()
        print('collecting inputs')
        if not isinstance(inputs, tuple):
            inputs = tuple(inputs)
        print('collecting params')
        for name, params in model.named_parameters():
            params_name.append(name)
            params_value_dict[name]=params.detach()
            params_shape.append(params.shape)
            params_dtype.append(params.dtype)
        return model, inputs, params_name, params_shape, params_dtype, params_value_dict, input_shapes
    else:
        return None

def trace_with_concrete_args(model, inputs, placeholder_num):
    concrete_args={}
    for _, (name, inp) in enumerate(zip(inspect.signature(model.forward).parameters.keys(), inputs)):
        concrete_args[name] = inp
        placeholder_num+=1
    traced = symbolic_trace(model, concrete_args=concrete_args)
    del concrete_args
    return traced, placeholder_num

def analyze_pytorch_model_with_new_shapes(model_name, model, inputs, original_params_name, original_params_shape, original_params_dtype):
    ops = []
    inputs_name, inputs_shape, inputs_dtype = [],[],[]
    params_name, params_shape, params_dtype=[],[],[]
    constant_name, constant_shape, constant_dtype = [],[],[]
    constant_value_dict = {}
    outputs_name, outputs_shape, outputs_dtype = [],[],[]
    intermediate_names, intermediate_shapes, intermediate_dtypes = [],[],[]
    name_start_idx=0
    placeholder_num=0
    print("Start tracing graph.")
    try:
        traced = symbolic_trace(model)
    except Exception as e:
        try:
            print(f"Error during symbolic tracing: {e}")
            print("Retrying with concrete arguments.")
            traced, placeholder_num=trace_with_concrete_args(model, inputs, placeholder_num)
        except Exception as e:
            print(f"Error during tracing with concrete arguments: {e}")
            print("cannot track")
            return False
    ShapeProp(traced).propagate(*inputs)
    print("Traced graph done.")
    for node in traced.graph.nodes:
        placeholder_num, ops, name_start_idx, inputs_name, inputs_shape, inputs_dtype, outputs_name, outputs_shape, outputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, constant_name, constant_shape, constant_dtype, constant_value_dict, original_params_name = handle_each_node(traced, placeholder_num, node, ops, name_start_idx, inputs_name, inputs_shape, inputs_dtype, outputs_name, outputs_shape, outputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, constant_name, constant_shape, constant_dtype, constant_value_dict, original_params_name, original_params_shape, original_params_dtype)
    print("handle each node done.")
    ops, inputs_name, outputs_name, intermediate_names, constant_name, params_name, name_start_idx, paramsname_constant_mapping = rename(ops, inputs_name, outputs_name, intermediate_names, constant_name, params_name, name_start_idx)
    # print(f'outputs_name:{outputs_name}, intermediate_names:{intermediate_names}')
    # print(f'outputs_dtype:{outputs_dtype}, intermediate_dtypes:{intermediate_dtypes}')
    ops=check_min_max(ops)
    ops,intermediate_dtypes,outputs_dtype=check_size(ops,intermediate_names, intermediate_dtypes,outputs_name, outputs_dtype)
    return ops, [inputs_name, inputs_shape, inputs_dtype], [outputs_name, outputs_shape, outputs_dtype], [intermediate_names,intermediate_shapes, intermediate_dtypes], [constant_name, constant_shape, constant_dtype], [params_name, params_shape, params_dtype], name_start_idx, paramsname_constant_mapping, constant_value_dict

def mapping_constant_params_value(params_value_dict, constant_value_dict, paramsname_constant_mapping, constant_names, params_names):
    constant_params_value=[]
    for constant_name in constant_names:
        real_constant_name=paramsname_constant_mapping[constant_name]
        constant_params_value.append(constant_value_dict[real_constant_name])
    for params_name in params_names:
        real_params_name=paramsname_constant_mapping[params_name]
        constant_params_value.append(params_value_dict[real_params_name])
    return constant_params_value

def get_inter_info(ir, known_names):
    #obtain intermediate names
    inter_names = list(set(re.findall(r'(?![LPVBU])[A-Z][a-z]*', ir))-set(known_names))
    return inter_names

def delete_inter(ir, inter_name):
    ir_split_list=split_ir(ir)
    new_ir_split_list=[]
    for ir_split_idx in range(len(ir_split_list)):
        ir_split=ir_split_list[ir_split_idx]
        new_ir_split=ir_split
        if '['+inter_name+'^' in ir_split or ';'+inter_name+'^' in ir_split:
            start_index=ir_split.index(inter_name+'^')
            end_index=ir_split[start_index:].index(';')+start_index
            new_ir_split=ir_split[:start_index] + ir_split[end_index+1:]
        if '[];' in new_ir_split:
            bracket_start_idx=new_ir_split.index('[];')
            if '[' in new_ir_split[bracket_start_idx+3:]:
                new_ir_split=new_ir_split.replace('[];','')
            else:
                if '[' in new_ir_split[:bracket_start_idx]:
                    before_list=new_ir_split[:bracket_start_idx]
                    reversed_before_list=before_list[::-1]
                    last_bracket_idx=len(before_list)-1-reversed_before_list.index('[')
                    new_ir_split=new_ir_split[:last_bracket_idx]
                else:
                    new_ir_split=''
        new_ir_split_list.append(new_ir_split)
    ir=''.join(new_ir_split_list)
    return ir

def sub_delete_no_use_inter(ir,inter_names):
    no_delete=True
    for inter_name in inter_names:
        if ir.count('['+inter_name+'^')+ir.count(';'+inter_name+'^')==ir.count(inter_name+'^'):
            ir=delete_inter(ir, inter_name)
            no_delete=False
    return ir, no_delete

def delete_no_use_inter(ir,inputs_info, outputs_info,constant_info, params_info):
    known_names, _, _, _=generate_IR_related_info(inputs_info, constant_info, params_info, outputs_info)
    inter_names=get_inter_info(ir, known_names)
    no_delete=False
    while not no_delete:
        ir, no_delete=sub_delete_no_use_inter(ir, inter_names)
        inter_names=get_inter_info(ir, known_names)
    return ir

def convert_to_IR(ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx):
    IR=''
    last_mean_info=[]
    len_ops = len(ops)
    # print(f'len_ops:{len_ops}, ops:{ops}')
    has_min_max=False
    for op in ops:
        IR,name_start_idx, last_mean_info, len_ops, has_min_max=handle_each_op(IR, op, len_ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx,last_mean_info, has_min_max)
    if len_ops==0:
        print(f'before deleting:{IR}')
        IR=delete_no_use_inter(IR,inputs_info, outputs_info,constant_info, params_info)
        print("IR done.")
        return IR, name_start_idx
    else:
        print("Error: not all ops are handled, len_ops:", len_ops)

def generate_IR_related_info(inputs_info, constant_info, params_info, outputs_info):
    known_names= inputs_info[0] + constant_info[0] + params_info[0] + outputs_info[0]
    known_shapes = inputs_info[1] + constant_info[1] + params_info[1] + outputs_info[1]
    known_dtype = inputs_info[2] + constant_info[2] + params_info[2] + outputs_info[2]
    input_known_names= inputs_info[0]+ constant_info[0] + params_info[0]
    return known_names, known_shapes, known_dtype, input_known_names

def apply_strategy_to_IR(IR, name_start_idx, strategy_name, input_output_name):
    row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops,eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops = split_IR_to_equations(IR)
    # print("loops:", loops)
    # print("equations_under_loops:", equations_under_loops)
    # print("eq_outputs_under_loops:", eq_outputs_under_loops)
    # print("eq_inputs_under_loops:", eq_inputs_under_loops)
    # print("simplified_eqs_under_loops:", simplified_eqs_under_loops)
    # print("simplified_eq_outputs_under_loops:", simplified_eq_outputs_under_loops)
    # print("simplified_eq_inputs_under_loops:", simplified_eq_inputs_under_loops)
    apply_strategy_to_IR = getattr(importlib.import_module("strategies"), f"apply_{strategy_name}_to_IR", None)
    original_IR_list, transformed_IR_list, has_transformation=apply_strategy_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops,eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    return original_IR_list, transformed_IR_list, has_transformation

def mapping_params_to_new_model(new_model, params_value_dict):
    for name, value in params_value_dict.items():
        if hasattr(new_model, name):
            param = getattr(new_model, name)
            if isinstance(param, torch.nn.Parameter):
                param.data.copy_(torch.tensor(value, dtype=param.dtype, device=param.device))
            else:
                setattr(new_model, name, torch.tensor(value, dtype=param.dtype, device=param.device))
        elif name in dict(new_model.named_parameters()):
            param = dict(new_model.named_parameters())[name]
            param.data.copy_(torch.tensor(value, dtype=param.dtype, device=param.device))
    return new_model

def check_two_outputs_precision_error(tvm_output, torch_output):
    torch_output_np = torch_output.cpu().detach().numpy()
    effective_elem=1-((np.isnan(torch_output_np)*np.isnan(tvm_output))+(np.isinf(torch_output_np)*np.isinf(tvm_output)))
    diff=np.abs(torch_output_np-tvm_output)
    atol=np.max(diff*effective_elem)
    rtol=np.max(diff*effective_elem/(np.abs(torch_output_np)+1e-8))
    MSE=np.mean((torch_output_np-tvm_output)**2*effective_elem)
    return atol, rtol, MSE
import copy
from .utils import *
import random
import math
from sympy import *
import re

# def judge_operator_fusion_condition_old(loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops):
#     #judge condition:
#     # 1. previous eq: the output E of the equation is not included in the input of this equation
#     # 2. there is at least one loops in this equation or in the previous equation
#     len_eqs = len(equations_under_loops)
#     start_index=0
#     start_index_1=0
#     has_loops= True
#     fusion_index_list=[]
#     for index in range(len_eqs):
#         # print(f"index: {index}, start_index: {start_index}")
#         vars=eq_outputs_under_loops[index]+ eq_inputs_under_loops[index]
#         if len(set(vars))<len(vars):
#             start_index=index+1
#             start_index_1=index+1
#             has_loops= True
#             continue
#         if index==start_index:
#             if loops[index]=='':
#                 has_loops= False
#         elif index>start_index:
#             if loops[index]!='':
#                 has_loops= True
#             if loops[index]=='' and not has_loops:
#                 continue
#             else:# enable fusion
#                 fusion_index_list.append([start_index, index])
#         if index>start_index_1 and start_index_1!=start_index:
#             if loops[index]=='' and not has_loops:
#                 continue
#             else:# enable fusion
#                 fusion_index_list.append([start_index_1, index])
#         start_index_1=index
#     return fusion_index_list

# def apply_tensor_split_to_decouple_operators_to_IR(IR, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
#     tensor_split_to_decouple_operators_index_list=judge_tensor_split_to_decouple_operators_condition(loops, eq_outputs_under_loops, simplified_eqs_under_loops)
#     has_transformation = False
#     transformed_IR_list=[]
#     for index in range(len(tensor_split_to_decouple_operators_index_list)):
#         has_transformation = True
#         op_index, eq_idx, split_axis, split_reduce_aix=tensor_split_to_decouple_operators_index_list[index]
#         print(f'op_index: {op_index}, eq_idx: {eq_idx}, split_axis:{split_axis}, split_reduce_aix:{split_reduce_aix}, simplified_eqs_under_loops:{simplified_eqs_under_loops[op_index]}')
#         this_loop = loops[op_index]
#         output_superscript = re.findall(r'\^\{.*?\}', eq_outputs_under_loops[index][0])[0]
#         output_subscript = re.findall(r'_{.*?}', eq_outputs_under_loops[op_index][0])[0]
#         inputs = eq_inputs_under_loops[op_index]
#         values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
#         #for split axis
#         s_axis=random.choice(split_axis)
#         num_cancat_inputs, inputs_index_list, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, loop_notation1, loop_notation2, new_values_list1=find_different_vars_scripts_and_curate_loops_for_tensor_split_decouple(s_axis, values_list[0], keys_list[0], loop_type_list[0], inputs)
#         intermediate_names, name_start_idx = generate_names(2, name_start_idx)
#         output_name=[intermediate_names[0],intermediate_names[1]]
#         new_input_name0, new_input_name1=[], []
#         for input_index in inputs_index_list:
#             new_input_name0.append(re.sub(r'\^\{.*?\}_{.*?}|\^\{.*?\}','',inputs[input_index]))
#             new_input_name1.append(re.sub(r'\^\{.*?\}_{.*?}|\^\{.*?\}','',inputs[input_index]))
#         if eq_outputs_under_loops[op_index][0] in eq_inputs_under_loops[op_index]:
#             output_index_in_input=eq_inputs_under_loops[op_index].index(eq_outputs_under_loops[op_index][0])
#             index_in_input_index_list=inputs_index_list.index(output_index_in_input)
#             new_input_name0[index_in_input_index_list]=intermediate_names[0]
#             new_input_name1[index_in_input_index_list]=intermediate_names[1]
#             output_resubscript = [inputs_subscripts_list[index_in_input_index_list],resubscript_list[index_in_input_index_list]]
#         else:
#             full_output_subscript=re.findall(r'[a-zA-Z]+|[^a-zA-Z]', output_subscript)
#             output_resubscript = [output_subscript, rewrite_subscript_for_cancat(full_output_subscript, [keys_list[0][s_axis]], [s_axis], new_values_list1)]
#         print(f'output_name: {output_name}, output_resubscript: {output_resubscript}')
#         #split and concat
#         concat_IR1=loop_notation1+'['+eq_outputs_under_loops[op_index][0]+'='+output_name[0]+output_superscript+output_resubscript[0]+';];'
#         concat_IR2=loop_notation2+'['+eq_outputs_under_loops[op_index][0]+'='+output_name[1]+output_superscript+output_resubscript[1]+';];'
#         #decouple the equation
#         expr1=transfrom_from_original_simpified_expr_to_modified_expr(simplified_eqs_under_loops[op_index][eq_idx].split('=')[1],simplified_eq_inputs_under_loops[op_index], eq_inputs_under_loops[op_index], inputs_index_list, new_input_name0, inputs_superscripts_list, inputs_subscripts_list)
#         expr2=transfrom_from_original_simpified_expr_to_modified_expr(simplified_eqs_under_loops[op_index][eq_idx].split('=')[1],simplified_eq_inputs_under_loops[op_index], eq_inputs_under_loops[op_index], inputs_index_list, new_input_name1, inputs_superscripts_list, resubscript_list)
#         decouple_IR1=loop_notation1+'['+output_name[0]+output_superscript+output_subscript+'='+expr1+';];'
#         decouple_IR2=loop_notation2+'['+output_name[1]+output_superscript+output_subscript+'='+expr2+';];'
#         transformed_part = decouple_IR1 + decouple_IR2 + concat_IR1 + concat_IR2 
#         transform_IR= ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
#         print(f'transformed_part: {transformed_part}')
#         transformed_IR_list.append(transform_IR)
#         #for reduce axis
#         if len(split_reduce_aix)>0:
#             r_axis=random.choice(split_reduce_aix)
#             r_value1, r_value2=random_split_a_loop(values_list[0][r_axis])
#     return [], False

def judge_operator_fusion_condition(loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops):
    #judge condition:
    # 1. previous eq: the output E of the equation is not included in the input of this equation
    # 2. there is at least one loops in this equation or in the previous equation
    len_eqs = len(equations_under_loops)
    start_index=0
    has_loops= True
    fusion_index_list=[]
    for index in range(len_eqs):
        # print(f"index: {index}, start_index: {start_index}")
        vars=eq_outputs_under_loops[index]+ eq_inputs_under_loops[index]
        if len(set(vars))<len(vars):
            start_index=index+1
            has_loops= True
            continue
        if index==start_index:
            if loops[index]=='':
                has_loops= False
        elif index>start_index:
            if loops[index]!='':
                has_loops= True
            if loops[index]=='' and not has_loops:
                continue
            else:# enable fusion
                full_output_in_first_eq=eq_outputs_under_loops[start_index]
                half_output_in_first_eq=[item[:item.index('}')+1] for item in full_output_in_first_eq]
                full_input_in_second_eq=eq_inputs_under_loops[index]+eq_outputs_under_loops[index]
                add_fusion=True
                for output_idx in range(len(half_output_in_first_eq)):
                    repeated_input_list=[item for item in full_input_in_second_eq if half_output_in_first_eq[output_idx] in item]
                    for repeated_input in repeated_input_list:
                        if repeated_input!=full_output_in_first_eq[output_idx]:
                            add_fusion=False
                            break
                    if not add_fusion:
                        break
                if add_fusion:
                    fusion_index_list.append([start_index, index])
        start_index=index
    return fusion_index_list

def judge_operator_fission_condition(loops, equations_under_loops, simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops):
    #1. loops exist;
    #2. multiple equations under the same loop;
    len_eqs = len(equations_under_loops)
    fission_index_list=[]
    for index in range(len_eqs):
        if loops[index]=='':
            continue
        if len(equations_under_loops[index])>1:
            fission_index_list.append(index)
    return fission_index_list

def judge_compute_inline_condition(loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, row_equations_under_loops):
    #judge condition:
    # 1. previous eq: the output E of the equation is not included in the input of this equation
    # 2. the input of this equation is the output of the previous equation
    # 3. has the same loops/no loops
    len_eqs = len(equations_under_loops)
    compute_inline_index_list=[]
    for index in range(1, len_eqs):
        # print(f"index: {index}, start_index: {start_index}")
        last_outputs=eq_outputs_under_loops[index-1]
        half_last_output=last_outputs[0][:last_outputs[0].index('}')+1]
        later_IR=''.join(row_equations_under_loops[index+1:])
        # print(f'half_last_output:{half_last_output}, later_IR: {later_IR}')
        vars=last_outputs+ eq_inputs_under_loops[index-1]
        if len(set(vars))<len(vars):
            continue
        # print(f':eq_outputs_under_loops[index-1][0]:{eq_outputs_under_loops[index-1][0]}, eq_inputs_under_loops[index]:{eq_inputs_under_loops[index]}')
        if last_outputs[0] in eq_inputs_under_loops[index] and half_last_output not in later_IR:
            if loops[index]==loops[index-1] or loops[index-1]=='' or loops[index]=='':
                if len(equations_under_loops[index-1])==1:
                    compute_inline_index_list.append([index-1, index])
    return compute_inline_index_list

def judge_expression_splitting_condition(simplified_eqs_under_loops):
    #judge condition:
    # we can just see if () exists, and if not only abs/sqrt/min/max/exp/log/if_then_else exist
    len_eqs = len(simplified_eqs_under_loops)
    expression_splitting_index_list=[]
    for index in range(len_eqs):
        eqs= simplified_eqs_under_loops[index]
        eq_index=0
        if len(eqs)==1:
            eq=eqs[0]
            expr = '='.join(eq.split('=')[1:])
            # print(f'expr:{expr}')
            if re.sub(r'abs\(.*?\)', '', expr) != '' and\
                len(re.findall(r'abs\(.*?\)', expr))==1 and\
                  not check_if_any_special_function_in_equations(re.findall(r'abs\(.*?\)', expr)[0]):
                subexpr=extract_subsexpression('abs',expr)
                expression_splitting_index_list.append([index,eq_index, subexpr])
            elif re.sub(r'sqrt\(.*?\)', '', expr) != '' and\
                len(re.findall(r'sqrt\(.*?\)', expr))==1 and\
                not check_if_any_special_function_in_equations(re.findall(r'sqrt\(.*?\)', expr)[0]):
                subexpr=extract_subsexpression('sqrt',expr)
                expression_splitting_index_list.append([index,eq_index, subexpr])
            elif re.sub(r'min\(.*?\)', '', expr) != '' and\
                len(re.findall(r'min\(.*?\)', expr))==1 and\
                not check_if_any_special_function_in_equations(re.findall(r'min\(.*?\)', expr)[0]):
                subexpr=extract_subsexpression('min',expr)
                expression_splitting_index_list.append([index,eq_index, subexpr])
            elif re.sub(r'max\(.*?\)', '', expr) != '' and\
                len(re.findall(r'max\(.*?\)', expr))==1 and\
                not check_if_any_special_function_in_equations(re.findall(r'max\(.*?\)', expr)[0]):
                subexpr=extract_subsexpression('max',expr)
                expression_splitting_index_list.append([index,eq_index, subexpr])
            elif re.sub(r'exp\(.*?\)', '', expr) != '' and\
                len(re.findall(r'exp\(.*?\)', expr))==1 and\
                not check_if_any_special_function_in_equations(re.findall(r'exp\(.*?\)', expr)[0]):
                subexpr=extract_subsexpression('exp',expr)
                expression_splitting_index_list.append([index,eq_index, subexpr])
            elif re.sub(r'log\(.*?\)', '', expr) != '' and\
                len(re.findall(r'log\(.*?\)', expr))==1 and\
                not check_if_any_special_function_in_equations(re.findall(r'log\(.*?\)', expr)[0]):
                subexpr=extract_subsexpression('log',expr)
                expression_splitting_index_list.append([index,eq_index, subexpr])
            elif re.sub(r'if_then_else\(.*?\)', '', expr) != '' and\
                len(re.findall(r'if_then_else\(.*?\)', expr))==1 and\
                not check_if_any_special_function_in_equations(re.findall(r'if_then_else\(.*?\)', expr)[0]):
                subexpr=extract_subsexpression('if_then_else',expr)
                expression_splitting_index_list.append([index,eq_index, subexpr])
            elif re.sub(r'\+.*?\+', '', expr) != '' and\
                len(re.findall(r'\+.*?\+', expr))==1 and\
                not check_if_any_special_function_in_equations(re.findall(r'\+.*?\+', expr)[0].replace('+', '')):
                subexpr=re.findall(r'\+.*?\+', expr)[0].replace('+', '')
                # print(f'here1:{subexpr}')
                expression_splitting_index_list.append([index,eq_index, subexpr])
            elif re.sub(r'\+.*?\*[a-zA-Z]+', '', expr) != '' and\
                len(re.findall(r'\+.*?\*[a-zA-Z]+', expr))==1 and\
                not check_if_any_special_function_in_equations(re.findall(r'\+.*?\*[a-zA-Z]+', expr)[0].replace('+', '')):
                subexpr=re.findall(r'\+.*?\*[a-zA-Z]+', expr)[0].replace('+', '')
                # print(f'here2:{subexpr}')
                expression_splitting_index_list.append([index,eq_index, subexpr])
            eq_index+=1
    return expression_splitting_index_list

def judge_tensor_concat_to_fuse_operators_condition(loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    # judge condition:
    # this equation has the same calculation as the previous equation.
    # these two equations has the same number of the loops
    # But these two equations are not the split or getitem (we do not check the types, e.g.,  A=C)
    # the output of the previous equation is not the input of this equation
    len_eqs = len(simplified_eqs_under_loops)
    tensor_concat_to_fuse_operators_index_list = []
    for index in range(1, len_eqs):
        last_output = simplified_eq_outputs_under_loops[index-1]
        _, _, _, last_len_list=split_loops_into_value_and_index([loops[index-1]])
        _, _, _, this_len_list=split_loops_into_value_and_index([loops[index]])
        if len(last_len_list)!=len(this_len_list) or (len(last_len_list)==1 and len(this_len_list)==1 and last_len_list[0] != this_len_list[0]):
            # print(f'last_len_list: {last_len_list}, this_len_list: {this_len_list}')
            continue
        if last_output[0] in simplified_eq_inputs_under_loops[index]:
            # print(f'last_output: {last_output}, simplified_eq_inputs_under_loops[index]: {simplified_eq_inputs_under_loops[index]}')
            continue
        if len(simplified_eqs_under_loops[index-1])==1 and len(simplified_eqs_under_loops[index])==1:
            last_eq = simplified_eqs_under_loops[index-1][0]
            last_eq_cal = re.sub(r'\b(?!max\b)(?!min\b)(?!exp\b)(?!log\b)(?!sqrt\b)(?!abs\b)(?!if_then_else\b)[a-zA-Z]+\b','',last_eq)
            if last_eq_cal =='=' or 'max' in last_eq_cal or 'min' in last_eq_cal or 'if_then_else' in last_eq_cal:
                continue
            this_eq = simplified_eqs_under_loops[index][0]
            this_eq_cal = re.sub(r'\b(?!max\b)(?!min\b)(?!exp\b)(?!log\b)(?!sqrt\b)(?!abs\b)(?!if_then_else\b)[a-zA-Z]+\b','',this_eq)
            # print(f'last_eq_cal: {last_eq_cal}, this_eq_cal: {this_eq_cal}')
            if last_eq_cal == this_eq_cal:
                tensor_concat_to_fuse_operators_index_list.append([index, 0, 0])
    return tensor_concat_to_fuse_operators_index_list

def judge_tensor_split_to_decouple_operators_condition(loops, simplified_eq_outputs_under_loops,simplified_eq_inputs_under_loops, eq_outputs_under_loops,eq_inputs_under_loops, simplified_eqs_under_loops):
    #condition: at least has one loop; 
    # the value of loop is more than 1;
    # the loop axis is not the reduce axis
    len_eqs = len(loops)
    tensor_split_to_decouple_operators_index_list = []
    for index in range(len_eqs):
        if loops[index] == '':
            continue
        if len(simplified_eqs_under_loops[index])==1:
            values_list, keys_list, _, _ = split_loops_into_value_and_index([loops[index]])
            # print(f'eq_outputs_under_loops[index][0]:{eq_outputs_under_loops[index]}')
            temp_previous_simplified_output = simplified_eq_outputs_under_loops[:index]
            previous_simplified_output=[iitem for item in temp_previous_simplified_output for iitem in item]
            this_simplified_input=simplified_eq_inputs_under_loops[index]
            # print(f'previous_simplified_output:{previous_simplified_output}, this_simplified_input: {this_simplified_input}')
            output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[index][0])
            output_subscript = output_subscript_list[0] if len(output_subscript_list)>0 else ''
            output_subscript_keys = re.findall(r'[a-z]+', output_subscript)
            split_axis=[idx for idx in range(len(values_list[0])) if (keys_list[0][idx] in output_subscript_keys) and (values_list[0][idx] > 1)]
            split_reduce_aix=[idx for idx in range(len(values_list[0])) if (keys_list[0][idx] not in output_subscript_keys) and (values_list[0][idx] > 1) and keys_list[0][idx]!='tx']
            # print(f'split_axis:{split_axis}, split_reduce_aix:{split_reduce_aix}')
            if len(split_axis)+len(split_reduce_aix)==0 or len(previous_simplified_output+this_simplified_input)>len(set(previous_simplified_output+this_simplified_input)):
                # print("here")
                continue
            # print(f'simplified_eqs_under_loops[index]: {simplified_eqs_under_loops[index]}')
            eq = simplified_eqs_under_loops[index][0]
            eq_cal = re.sub(r'\b(?!max\b)(?!min\b)(?!exp\b)(?!log\b)(?!sqrt\b)(?!abs\b)(?!if_then_else\b)[a-zA-Z]+\b','',eq)
            if eq_cal == '=':
                # print(f'eq_cal: {eq_cal}, continue')
                continue
            else:
                tensor_split_to_decouple_operators_index_list.append([index, 0, split_axis, split_reduce_aix])
    return tensor_split_to_decouple_operators_index_list

def judge_common_subexpression_elimination_condition(loops, simplified_eqs_under_loops, eq_outputs_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops):
    # judge condition: find the subexpression
    len_eqs = len(loops)
    common_subexpression_elimination_index_mapping = {}
    subexpression_info={}
    pattern_list=[r'exp\(.*?\)', r'log\(.*?\)', r'sqrt\(.*?\)', r'abs\(.*?\)', r'min\(.*?\)', r'max\(.*?\)', r'if_then_else\(.*?\)',r'[a-zA-Z]+\*[a-zA-Z]+']
    for index in range(len_eqs):
        for eq_index in range(len(simplified_eqs_under_loops[index])):
            simplified_eq= simplified_eqs_under_loops[index][eq_index]
            simplified_cal = '='.join(simplified_eq.split('=')[1:])
            # print(f'simplified_eq:{simplified_eq},simplified_cal: {simplified_cal}')
            simplified_cal_remove_output=simplified_cal.replace(simplified_eq.split('=')[0], '')
            for pattern in pattern_list:
                subexpression_info=check_subexpression(pattern, simplified_cal_remove_output, subexpression_info, [index, eq_index], simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    for key, values in subexpression_info.items():
        if len(values)>1:
            for value in values:
                if str(value) not in common_subexpression_elimination_index_mapping.keys():
                    common_subexpression_elimination_index_mapping[str(value)] = [key]
                else:
                    if key not in common_subexpression_elimination_index_mapping[str(value)]:
                        common_subexpression_elimination_index_mapping[str(value)].append(key)
    return common_subexpression_elimination_index_mapping

def judge_dead_code_elimination_condition(simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    # judge condition: the output of the equation is not used in any other equations
    len_eqs = len(simplified_eq_outputs_under_loops)
    dead_code_elimination_index_list = []
    for index in range(len_eqs-1):
        later_inputs = sum(simplified_eq_inputs_under_loops[index+1:], [])
        this_outputs = simplified_eq_outputs_under_loops[index]
        output_not_in_later_inputs = True
        for this_output in this_outputs:
            if this_output in later_inputs:
                output_not_in_later_inputs= False  
        if output_not_in_later_inputs:
            dead_code_elimination_index_list.append(index)
    return dead_code_elimination_index_list

def judge_expression_reorder_condition(simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    #judge condition: the output of the equation is not used in the next equations
    len_eqs = len(simplified_eq_outputs_under_loops)
    expression_reorder_index_list = []
    for index in range(len_eqs-1):
        this_outputs = simplified_eq_outputs_under_loops[index]
        next_inputs = simplified_eq_inputs_under_loops[index+1]
        this_output_not_in_next_inputs = True
        for this_output in this_outputs:
            if this_output in next_inputs:
                this_output_not_in_next_inputs=False
        if this_output_not_in_next_inputs:
            expression_reorder_index_list.append(index)
    return expression_reorder_index_list

def judge_loop_reorder_condition(loops):
    #two loops exist
    len_loops = len(loops)
    loop_reorder_index_list = []
    for index in range(len_loops):
        this_loop = loops[index]
        _, _, _, len_list = split_loops_into_value_and_index([this_loop])
        if len_list[0]>2:
            loop_reorder_index_list.append(index)
    return loop_reorder_index_list

# def judge_loop_tiling_condition(loops):
#     #two loops exist and the values can be divided into two parts
#     len_loops = len(loops)
#     loop_tiling_index_list = []
#     for index in range(len_loops):
#         this_loop = loops[index]
#         values_list, keys_list, _, len_list = split_loops_into_value_and_index([this_loop])
#         if len_list[0]>2:
#             bx_index = keys_list[0].index('bx')
#             for loop_idx in range(len(values_list[0])-1):
#                 if loop_idx!=bx_index:
#                     if loop_idx==bx_index-1:
#                         this_value = factorization(values_list[0][loop_idx])
#                         next_value = factorization(values_list[0][loop_idx+2])
#                     else:
#                         this_value = factorization(values_list[0][loop_idx])
#                         next_value = factorization(values_list[0][loop_idx+1])
#                     if len(this_value) > 1 and len(next_value) > 1:
#                         loop_tiling_index_list.append([index, loop_idx, this_value[1:], next_value[1:]])
#     return loop_tiling_index_list

def judge_loop_tiling_condition(loops,eq_outputs_under_loops,simplified_eqs_under_loops):
    #two loops exist and the values can be divided into two parts
    len_loops = len(loops)
    loop_tiling_index_list = []
    for index in range(len_loops):
        this_loop = loops[index]
        this_output= eq_outputs_under_loops[index][0]
        this_simplified_eqs= simplified_eqs_under_loops[index]
        no_var=True
        for simplified_eq in this_simplified_eqs:
            simplified_eq=simplified_eq.replace('max','').replace('min','').replace('exp','').replace('log','').replace('sqrt','').replace('abs','').replace('if_then_else','').replace('erf','')
            if len(re.findall(rf'\b[a-z]+\b',simplified_eq))>0:
                no_var=False
                break
        this_output_subscript_list, _ = find_subscripts_of_input_output_and_simplified_version(this_output)
        this_output_subscript = this_output_subscript_list[0] if len(this_output_subscript_list)>0 else ''
        this_output_subscript_keys = re.findall(r'[a-z]+', this_output_subscript)
        values_list, keys_list, _, len_list = split_loops_into_value_and_index([this_loop])
        if len_list[0]>2 and no_var:
            for loop_idx in range(len(values_list[0])-1):
                this_key= keys_list[0][loop_idx]
                next_key= keys_list[0][loop_idx+1]
                if this_key in this_output_subscript_keys and next_key in this_output_subscript_keys:
                    this_value = factorization(values_list[0][loop_idx])
                    next_value = factorization(values_list[0][loop_idx+1])
                    if len(this_value) > 1 and len(next_value) > 1:
                        loop_tiling_index_list.append([index, loop_idx, this_value[1:], next_value[1:]])
    return loop_tiling_index_list

def judge_loop_split_condition(loops,eq_outputs_under_loops,simplified_eqs_under_loops):
    len_loops = len(loops)
    loop_split_index_list = []
    for index in range(len_loops):
        this_loop = loops[index]
        this_output= eq_outputs_under_loops[index][0]
        this_simplified_eqs= simplified_eqs_under_loops[index]
        no_var=True
        for simplified_eq in this_simplified_eqs:
            simplified_eq=simplified_eq.replace('max','').replace('min','').replace('exp','').replace('log','').replace('sqrt','').replace('abs','').replace('if_then_else','').replace('erf','')
            if len(re.findall(rf'\b[a-z]+\b',simplified_eq))>0:
                no_var=False
                break
        this_output_subscript_list, _ = find_subscripts_of_input_output_and_simplified_version(this_output)
        this_output_subscript = this_output_subscript_list[0] if len(this_output_subscript_list)>0 else ''
        this_output_subscript_keys = re.findall(r'[a-z]+', this_output_subscript)
        values_list, keys_list, _, len_list = split_loops_into_value_and_index([this_loop])
        if len_list[0]>1 and no_var:
            for loop_idx in range(len(values_list[0])):
                this_value = factorization(values_list[0][loop_idx])
                this_key= keys_list[0][loop_idx]
                if len(this_value) > 1 and this_key in this_output_subscript_keys:
                    loop_split_index_list.append([index, loop_idx, this_value[1:]])
    return loop_split_index_list

def judge_loop_unrolling_condition(loops):
    # one loop exists and this loop is not the binding loop
    len_loops = len(loops)
    loop_unrolling_index_list = []
    for index in range(len_loops):
        this_loop = loops[index]
        _, _, loop_type_list, len_list = split_loops_into_value_and_index([this_loop])
        if len_list[0]>2:
            if 'L' in loop_type_list[0]:
                loop_unrolling_index_list.append(index)
    return loop_unrolling_index_list

def judge_loop_parallelization_condition(loops, eq_outputs_under_loops):
    # one loop exists and this loop is not the binding loop and is not the reduce axis
    len_loops = len(loops)
    loop_parallelization_index_list = []
    for index in range(len_loops):
        this_loop = loops[index]
        _, keys_list, loop_type_list, len_list = split_loops_into_value_and_index([this_loop])
        if len_list[0]>2:
            if 'L' in loop_type_list[0]:
                reduce_axis=[]
                for idx in range(len(eq_outputs_under_loops[index])):
                    output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[index][idx])
                    output_subscript = output_subscript_list[0] if len(output_subscript_list)>0 else ''
                    output_subscript_keys = re.findall(r'[a-z]+', output_subscript)
                    reduce_axis.extend([item for item in keys_list[0] if item not in output_subscript_keys])
                parallel_axis=list(set(keys_list[0])- set(reduce_axis+['tx']))
                if len(parallel_axis)>0:
                    loop_parallelization_index_list.append([index, parallel_axis])
    return loop_parallelization_index_list

# def judge_loop_binding_condition(loops, eq_outputs_under_loops):
#     # one loop exists and this loop is not the binding loop and is not the reduce axis
#     # and can be factorization
#     len_loops = len(loops)
#     loop_binding_index_list = []
#     for index in range(len_loops):
#         this_loop = loops[index]
#         values_list, keys_list, _, len_list = split_loops_into_value_and_index([this_loop])
#         if len_list[0]>2:
#             if 'bx' in keys_list[0]:
#                 bx_index = keys_list[0].index('bx')
#             else:
#                 bx_index=None
#             tx_index = keys_list[0].index('tx')
#             reduce_axis=[]
#             for idx in range(len(eq_outputs_under_loops[index])):
#                 output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[index][idx])
#                 output_subscript = output_subscript_list[0] if len(output_subscript_list)>0 else ''
#                 output_subscript_keys = re.findall(r'[a-z]+', output_subscript)
#                 reduce_axis.extend([item for item in keys_list[0] if item not in output_subscript_keys])
#             for loop_idx in range(len(values_list[0])):
#                 if loop_idx!=bx_index and loop_idx!=tx_index and keys_list[0][loop_idx] not in reduce_axis:
#                     this_value = factorization(values_list[0][loop_idx])
#                     if len(this_value) > 0:
#                         loop_binding_index_list.append([index, loop_idx, this_value])
#     return loop_binding_index_list

def judge_reduction_factorization_condition(loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops, simplified_eqs_under_loops):
    #condition: at least has one loop; 
    # the value of loop is more than 1;
    # the loop axis is the reduce axis
    reduction_factorization_index_list = []
    len_eqs = len(loops)
    for index in range(len_eqs):
        if loops[index] == '':
            continue
        if len(simplified_eqs_under_loops[index])==1:
            # this_simplified_output = simplified_eq_outputs_under_loops[index][0]
            # updated_input=[input_item for input_item in eq_inputs_under_loops[index] if this_simplified_output+'^' not in input_item]
            temp_previous_simplified_output = simplified_eq_outputs_under_loops[:index]
            previous_simplified_output=[iitem for item in temp_previous_simplified_output for iitem in item]
            this_simplified_input=simplified_eq_inputs_under_loops[index]
            values_list, keys_list, _, _ = split_loops_into_value_and_index([loops[index]])
            output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[index][0])
            output_subscript = output_subscript_list[0] if len(output_subscript_list)>0 else ''
            output_subscript_keys = re.findall(r'[a-z]+', output_subscript)
            # print(f'output_subscript_keys:{output_subscript_keys},values_list:{values_list[0]}, keys_list:{keys_list[0]}')
            split_reduce_aix=[idx for idx in range(len(values_list[0])) if (keys_list[0][idx] not in output_subscript_keys) and (values_list[0][idx] > 1) and keys_list[0][idx]!='tx']
            # print(f'split_reduce_aix:{split_reduce_aix}')
            if len(split_reduce_aix)==0 or len(previous_simplified_output+this_simplified_input)>len(set(previous_simplified_output+this_simplified_input)):
                # print("here")
                continue
            eq = simplified_eqs_under_loops[index][0]
            eq_cal = re.sub(r'\b(?!max\b)(?!min\b)(?!exp\b)(?!log\b)(?!sqrt\b)(?!abs\b)(?!if_then_else\b)[a-zA-Z]+\b','',eq)
            if eq_cal == '=':
                continue
            else:
                reduction_factorization_index_list.append([index, 0, split_reduce_aix])
    return reduction_factorization_index_list

def judge_cache_read_write_condition(simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops, simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops):
    cache_read_write_index_list = []
    len_eqs = len(eq_outputs_under_loops)
    for index in range(len_eqs):
        this_inputs = eq_inputs_under_loops[index]
        temp_previous_simplified_output = simplified_eq_outputs_under_loops[:index]
        previous_simplified_output=[iitem for item in temp_previous_simplified_output for iitem in item]
        this_simplified_input=simplified_eq_inputs_under_loops[index]
        if len(simplified_eqs_under_loops[index])==1 and len(previous_simplified_output+this_simplified_input)==len(set(previous_simplified_output+this_simplified_input)):
            this_output =  eq_outputs_under_loops[index][0]
            can_be_selected_inputs=[]
            for this_input in this_inputs:
                if this_input!=this_output:
                    this_input_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_input)
                    if len(this_input_subscript_list)>0 and len(re.findall(r'[a-zA-Z]+',''.join(this_input_subscript_list)))>0:
                        can_be_selected_inputs.append(this_input)
            this_output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_output)
            if len(this_output_subscript_list)>0:
                cache_read_write_index_list.append([index, can_be_selected_inputs, [this_output]])
            else:
                cache_read_write_index_list.append([index, can_be_selected_inputs, []])
    return  cache_read_write_index_list

def judge_layout_transformation_condition(loops, simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops):
    #transformation types: 
    # 1.transpose(multiple dims(>1)->multiple dims, same number of dim) 
    # 2. reshape(multiple dims/single dim(>=1, factorization)->more dims, multiple dims (>1)->less dims)
    # 3. flatten (multiple dims(>1)->single dim)
    # 4. squeeze (multiple dims including 1(>1)-> less dims)
    # eq: cannot be A=B
    # this input cannot also be the output
    layout_transformation_index_list=[]
    len_eqs = len(eq_inputs_under_loops)
    for index in range(len_eqs):
        this_inputs = eq_inputs_under_loops[index]
        this_output = eq_outputs_under_loops[index][0]
        this_eqs = simplified_eqs_under_loops[index]
        if len(this_eqs)>1 or re.sub(r'[a-zA-Z]+','','='.join(this_eqs[0].split('=')[1:]))=='':
            continue
        candidate_inputs_transpose_flatten_reshape2=[]
        candidate_inputs_reshape1=[]
        candidate_inputs_squeeze=[]
        for this_input in this_inputs:
            if this_input!=this_output:
                #find subscript
                this_input_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_input)
                #check transformation list
                if len(this_input_subscript_list)>0:
                    subscript_details=generate_subscript_details(this_input_subscript_list)
                    # print(f'this_input:{this_input}\nthis_input_subscript_list:{this_input_subscript_list}\nsubscript_details: {subscript_details}')
                    remove_bx_and_var_subscript_details=[item for item in subscript_details if item!='bx' and len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))==0]
                    var_subscript_details=[item for item in subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))>0]
                    remove_num_subscript_details=[item for item in subscript_details if not item.isdigit()]
                    if len(remove_bx_and_var_subscript_details)>1 and len(var_subscript_details)==0 and len(remove_num_subscript_details)>0:
                        candidate_inputs_transpose_flatten_reshape2.append([this_input, subscript_details])
                        if len(subscript_details)-len(remove_num_subscript_details)>0:
                            num_subscript_idx_in_subscript_details= [idx for idx in range(len(subscript_details)) if subscript_details[idx].isdigit()]
                            candidate_inputs_squeeze.append([this_input, subscript_details, num_subscript_idx_in_subscript_details])
                    values_list, keys_list, _, _ = split_loops_into_value_and_index([loops[index]])
                    key_value_mapping={keys_list[0][idx]: values_list[0][idx]-1 for idx in range(len(keys_list[0]))}
                    if len(var_subscript_details)==0:
                        subscript_details_value =[eval(expr, {}, key_value_mapping)+1 for expr in subscript_details]
                    for subscript_idx in range(len(subscript_details)):
                        subscript=subscript_details[subscript_idx]
                        # this_output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_output)
                        # output_subscript_details=re.findall(rf'[a-z]+',this_output_subscript_list[0]) if len(this_output_subscript_list)>0 else []
                        if len(var_subscript_details)==0:# and subscript not in output_subscript_details:
                            subscript_value= subscript_details_value[subscript_idx]
                            this_value = factorization(subscript_value)
                            if len(this_value)>1 and re.sub(r'[a-z]+','',subscript)=='':  # multiple dims(>1)->single dim
                                candidate_inputs_reshape1.append([this_input, subscript_details, subscript_details_value, subscript_idx, this_value[1:]])
        layout_transformation_index_list.append([index, candidate_inputs_transpose_flatten_reshape2, candidate_inputs_reshape1, candidate_inputs_squeeze])
    return layout_transformation_index_list

def judge_memory_coalescing_condition(loops, simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops):
    #index more than 1
    #index: has calculation
    len_eqs = len(eq_outputs_under_loops)
    memory_coalescing_index_list = []
    for index in range(len_eqs):
        if len(simplified_eqs_under_loops[index])==1:
            this_output = eq_outputs_under_loops[index][0]
            values_list, keys_list, _, _ = split_loops_into_value_and_index([loops[index]])
            key_value_mapping={keys_list[0][idx]: values_list[0][idx]-1 for idx in range(len(keys_list[0]))}
            this_output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_output)
            if len(this_output_subscript_list)>0:
                subscript_details=generate_subscript_details(this_output_subscript_list)
                var_subscript_details=[item for item in subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))>0]
                remove_num_subscript_details=[item for item in subscript_details if not item.isdigit()]
                coalescing_subscript_details=[item for item in subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))==0 and re.sub(r'[a-z]+','',item)!='']
                if len(var_subscript_details)==0 and len(remove_num_subscript_details)>1 and len(coalescing_subscript_details)>0:
                    subscript_details_value =[eval(expr, {}, key_value_mapping)+1 for expr in subscript_details]
                    memory_coalescing_index_list.append([index, this_output, subscript_details,subscript_details_value, coalescing_subscript_details])
            this_inputs = eq_inputs_under_loops[index]
            for this_input in this_inputs:
                if this_input!=this_output:
                    this_input_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_input)
                    if len(this_input_subscript_list)>0:
                        subscript_details=generate_subscript_details(this_input_subscript_list)
                        var_subscript_details=[item for item in subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))>0]
                        remove_num_subscript_details=[item for item in subscript_details if not item.isdigit()]
                        coalescing_subscript_details=[item for item in subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))==0 and re.sub(r'[a-z]+','',item)!='']
                        if len(var_subscript_details)==0 and len(remove_num_subscript_details)>1 and len(coalescing_subscript_details)>0:
                            subscript_details_value =[eval(expr, {}, key_value_mapping)+1 for expr in subscript_details]
                            memory_coalescing_index_list.append([index, this_input, subscript_details,subscript_details_value, coalescing_subscript_details])
    return memory_coalescing_index_list

def judge_vectorized_memory_access_condition(simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops):
    vectorized_memory_access_index_list = []
    len_eqs = len(eq_outputs_under_loops)
    for index in range(len_eqs):
        this_inputs = eq_inputs_under_loops[index]
        if len(simplified_eqs_under_loops[index])==1:
            this_output =  eq_outputs_under_loops[index][0]
            candidate_inputs=[]
            for this_input in this_inputs:
                if this_input!=this_output:
                    this_input_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_input)
                    if len(this_input_subscript_list)>0:
                        subscript_details=generate_subscript_details(this_input_subscript_list)
                        this_input_keys= re.findall(r'[a-zA-Z]+', this_input_subscript_list[0])
                        var_subscript_details=[item for item in subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))>0]
                        if len(subscript_details)>0 and len(var_subscript_details)==0:
                            if ('bx' in this_input_keys and len(subscript_details)>2) or ('bx' not in this_input_keys and len(subscript_details)>1):
                                candidate_inputs.append(this_input)
            this_output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_output)
            if len(this_output_subscript_list)>0:
                subscript_details=generate_subscript_details(this_output_subscript_list)
                this_output_keys= re.findall(r'[a-zA-Z]+', this_output_subscript_list[0])
                var_subscript_details=[item for item in subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))>0]
                if len(var_subscript_details)==0 and (('bx' in this_output_keys and len(subscript_details)>2) or ('bx' not in this_output_keys and len(subscript_details)>1)):
                    vectorized_memory_access_index_list.append([index, candidate_inputs, [this_output]])
                else:
                    vectorized_memory_access_index_list.append([index, candidate_inputs, []])
            else:
                vectorized_memory_access_index_list.append([index, candidate_inputs, []])
    return  vectorized_memory_access_index_list

def judge_set_storage_scope_condition(input_output_name, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    #if the input or output is not in the input_output_name
    set_storage_scope_index_mapping={}
    len_eqs = len(simplified_eq_outputs_under_loops)
    for index in range(len_eqs):
        this_inputs = simplified_eq_inputs_under_loops[index]
        this_outputs = simplified_eq_outputs_under_loops[index]
        for this_output in this_outputs:
            if this_output not in input_output_name:
                if this_output not in set_storage_scope_index_mapping.keys():
                    set_storage_scope_index_mapping[this_output] = [index]
                else:
                    set_storage_scope_index_mapping[this_output].append(index)
        for this_input in this_inputs:
            if this_input not in input_output_name and this_input not in this_outputs:
                if this_input not in set_storage_scope_index_mapping.keys():
                    set_storage_scope_index_mapping[this_input] = [index]
                else:
                    set_storage_scope_index_mapping[this_input].append(index)
    return set_storage_scope_index_mapping

def sub_set_storage_layout_condition(simplified_inputs_outputs_list, input_output_name, loops, this_input, index, candidate_inputs_transpose_flatten_reshape2,candidate_inputs_reshape1, candidate_inputs_squeeze):
    #find subscript
    this_input_subscript_list, this_simplified_input=find_subscripts_of_input_output_and_simplified_version(this_input)
    # print(f'this_input:{this_input},this_simplified_input:{this_simplified_input}, this_input_subscript_list: {this_input_subscript_list}')
    #check transformation list
    if len(this_input_subscript_list)>0 and this_simplified_input not in input_output_name: # and re.sub(r'[a-z]+','',this_input_subscript_list[0].replace('_{','').replace('}',''))=='':
        # print(f'this_input_subscript_list[0]:{this_input_subscript_list[0]}')
        subscript_details=generate_subscript_details(this_input_subscript_list)
        # print(f'this_input:{this_input}\nthis_input_subscript_list:{this_input_subscript_list}\nsubscript_details: {subscript_details}')
        remove_bx_and_var_subscript_details=[item for item in subscript_details if item!='bx' and len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))==0]
        var_subscript_details=[item for item in subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))>0]
        remove_num_subscript_details=[item for item in subscript_details if not item.isdigit()]
        if len(var_subscript_details)==0:
            if len(remove_bx_and_var_subscript_details)>1 and len(remove_num_subscript_details)>0:
                if this_simplified_input not in candidate_inputs_transpose_flatten_reshape2:
                    candidate_inputs_transpose_flatten_reshape2[this_simplified_input]=[[index, this_input, subscript_details]]
                else:
                    candidate_inputs_transpose_flatten_reshape2[this_simplified_input].append([index, this_input, subscript_details])
            values_list, keys_list, _, _ = split_loops_into_value_and_index([loops[index]])
            key_value_mapping={keys_list[0][idx]: values_list[0][idx]-1 for idx in range(len(keys_list[0]))}
            subscript_details_value =[eval(expr, {}, key_value_mapping)+1 for expr in subscript_details]
            for subscript_idx in range(len(subscript_details)):
                subscript=subscript_details[subscript_idx]
                subscript_value= subscript_details_value[subscript_idx]
                this_value = factorization(subscript_value)
                if len(this_value)>1 and re.sub(r'[a-z]+','',subscript)=='':  # multiple dims(>1)->single dim
                    if this_simplified_input not in candidate_inputs_reshape1:
                        candidate_inputs_reshape1[this_simplified_input]=[[index, this_input, subscript_details, subscript_details_value, subscript_idx, this_value[1:]]]
                    else:
                        candidate_inputs_reshape1[this_simplified_input].append([index, this_input, subscript_details, subscript_details_value, subscript_idx, this_value[1:]])
                if subscript_value==1:
                    if this_simplified_input not in candidate_inputs_squeeze:
                        candidate_inputs_squeeze[this_simplified_input]=[[index, this_input, subscript_details, subscript_idx]]
                    else:
                        has_equal=False
                        for past_idx in range(len(candidate_inputs_squeeze[this_simplified_input])):
                            if candidate_inputs_squeeze[this_simplified_input][past_idx][0]!=index and candidate_inputs_squeeze[this_simplified_input][past_idx][3]==subscript_idx:
                                has_equal=True
                                break
                            elif candidate_inputs_squeeze[this_simplified_input][past_idx][0]==index and candidate_inputs_squeeze[this_simplified_input][past_idx][3]!=subscript_idx:
                                has_equal=True
                                break
                        if has_equal:
                            candidate_inputs_squeeze[this_simplified_input].append([index, this_input, subscript_details, subscript_idx])
                        else:
                            candidate_inputs_squeeze.pop(this_simplified_input)
        #delete invalid transformation
        if len(subscript_details)-len(remove_num_subscript_details)>0:
            if this_simplified_input in candidate_inputs_transpose_flatten_reshape2:
                candidate_inputs_transpose_flatten_reshape2.pop(this_simplified_input)
            if this_simplified_input in candidate_inputs_reshape1:
                candidate_inputs_reshape1.pop(this_simplified_input)
            if this_simplified_input in candidate_inputs_squeeze:
                candidate_inputs_squeeze.pop(this_simplified_input)
        if len(var_subscript_details)>0:
            for var in var_subscript_details:
                this_simplified_var = re.sub(r'^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', var)
                if this_simplified_var in candidate_inputs_transpose_flatten_reshape2:
                    candidate_inputs_transpose_flatten_reshape2.pop(this_simplified_var)
                if this_simplified_var in candidate_inputs_reshape1:
                    candidate_inputs_reshape1.pop(this_simplified_var)
                if this_simplified_var in candidate_inputs_squeeze:
                    candidate_inputs_squeeze.pop(this_simplified_var)
    return candidate_inputs_transpose_flatten_reshape2, candidate_inputs_reshape1, candidate_inputs_squeeze

def judge_set_storage_layout_condition(input_output_name, loops,simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    #transformation types: 
    # 1.transpose(multiple dims(>1)->multiple dims, same number of dim) 
    # 2. reshape(multiple dims/single dim(>=1, factorization)->more dims, multiple dims (>1)->less dims)
    # 3. flatten (multiple dims(>1)->single dim)
    # 4. squeeze (multiple dims including 1(>1)-> less dims)
    # eq: cannot be A=B
    len_eqs = len(simplified_eqs_under_loops)
    simplified_inputs_outputs_list=[]
    for idx in range(len_eqs):
        simplified_inputs_outputs_list+=list(set(simplified_eq_outputs_under_loops[idx]+simplified_eq_inputs_under_loops[idx]))
    # print(f'simplified_inputs_outputs_list:{simplified_inputs_outputs_list}')
    candidate_inputs_transpose_flatten_reshape2={}
    candidate_inputs_reshape1={}
    candidate_inputs_squeeze={}
    for index in range(len_eqs):
        if len(simplified_eqs_under_loops[index])==1:
            this_output = eq_outputs_under_loops[index][0]
            candidate_inputs_transpose_flatten_reshape2, candidate_inputs_reshape1, candidate_inputs_squeeze= sub_set_storage_layout_condition(simplified_inputs_outputs_list, input_output_name, loops, this_output, index, candidate_inputs_transpose_flatten_reshape2, candidate_inputs_reshape1, candidate_inputs_squeeze)
            this_inputs = eq_inputs_under_loops[index]
            for this_input in this_inputs:
                candidate_inputs_transpose_flatten_reshape2, candidate_inputs_reshape1, candidate_inputs_squeeze= sub_set_storage_layout_condition(simplified_inputs_outputs_list, input_output_name, loops, this_input, index, candidate_inputs_transpose_flatten_reshape2, candidate_inputs_reshape1, candidate_inputs_squeeze)
    candidate_inputs_transpose_flatten_reshape2_keys=list(candidate_inputs_transpose_flatten_reshape2.keys())
    candidate_inputs_reshape1_keys=list(candidate_inputs_reshape1.keys())
    candidate_inputs_squeeze_keys=list(candidate_inputs_squeeze.keys())
    for this_simplified_input in candidate_inputs_transpose_flatten_reshape2_keys:
        transpose_flatten_reshape2_ops=[candidate_inputs_transpose_flatten_reshape2[this_simplified_input][idx][0] for idx in range(len(candidate_inputs_transpose_flatten_reshape2[this_simplified_input]))]
        if len(list(set(transpose_flatten_reshape2_ops)))<simplified_inputs_outputs_list.count(this_simplified_input):
            candidate_inputs_transpose_flatten_reshape2.pop(this_simplified_input)
    for this_simplified_input in candidate_inputs_reshape1_keys:
        reshape1_ops=[candidate_inputs_reshape1[this_simplified_input][idx][0] for idx in range(len(candidate_inputs_reshape1[this_simplified_input]))]
        if len(list(set(reshape1_ops)))<simplified_inputs_outputs_list.count(this_simplified_input):
            candidate_inputs_reshape1.pop(this_simplified_input)
    for this_simplified_input in candidate_inputs_squeeze_keys:
        squeeze_ops=[candidate_inputs_squeeze[this_simplified_input][idx][0] for idx in range(len(candidate_inputs_squeeze[this_simplified_input]))]
        if len(list(set(squeeze_ops)))<simplified_inputs_outputs_list.count(this_simplified_input):
            candidate_inputs_squeeze.pop(this_simplified_input)
    set_storage_layout_index_list=[candidate_inputs_transpose_flatten_reshape2, candidate_inputs_reshape1, candidate_inputs_squeeze]
    return set_storage_layout_index_list

def judge_set_storage_align_condition(loops, simplified_eqs_under_loops, input_output_name, eq_outputs_under_loops, simplified_eq_outputs_under_loops):
    set_storage_align_index_mapping = {}
    len_eqs = len(eq_outputs_under_loops)
    for index in range(len_eqs):
        # this_inputs = eq_inputs_under_loops[index]
        if len(simplified_eqs_under_loops[index])==1:
            this_simplified_output = simplified_eq_outputs_under_loops[index][0]
            this_output =  eq_outputs_under_loops[index][0]
            values_list, keys_list, _, _ = split_loops_into_value_and_index([loops[index]])
            key_value_mapping={keys_list[0][idx]: values_list[0][idx]-1 for idx in range(len(keys_list[0]))}
            if this_simplified_output not in input_output_name:
                this_output_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(this_output)
                if len(this_output_subscript_list)>0:
                    output_subscript_details=generate_subscript_details(this_output_subscript_list)
                    output_var_subscript_details=[item for item in output_subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))>0]
                    if len(output_var_subscript_details)==0:
                        output_subscript_details_value =[eval(expr, {}, key_value_mapping)+1 for expr in output_subscript_details]
                        if this_simplified_output not in set_storage_align_index_mapping.keys():
                            set_storage_align_index_mapping[this_simplified_output] = [index, [output_subscript_details_value]]
            # for this_input in this_inputs:
            #     if this_input!=this_output and this_input not in input_output_name:
            #         this_input_subscript_list,this_simplified_input=find_subscripts_of_input_output_and_simplified_version(this_input)
            #         if len(this_input_subscript_list)>0:
            #             input_subscript_details=generate_subscript_details(this_input_subscript_list)
            #             input_var_subscript_details=[item for item in input_subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))>0]
            #             if len(input_var_subscript_details)==0:
            #                 input_subscript_details_value =[eval(expr, {}, key_value_mapping)+1 for expr in input_subscript_details]
            #                 if this_simplified_input not in set_storage_align_index_mapping.keys():
            #                     set_storage_align_index_mapping[this_simplified_input] = [[input_subscript_details_value]]
    return  set_storage_align_index_mapping

def judge_asynchronous_pipeline_condition(simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops):
    asynchronous_pipeline_index_list=[]
    len_eqs = len(eq_outputs_under_loops)
    for index in range(len_eqs):
        this_inputs = eq_inputs_under_loops[index]
        if len(simplified_eqs_under_loops[index])==1:
            this_output =  eq_outputs_under_loops[index][0]
            candidate_inputs=[]
            for this_input in this_inputs:
                if this_input!=this_output:
                    this_input_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_input)
                    if len(this_input_subscript_list)>0:
                        candidate_inputs.append(this_input)
            if len(candidate_inputs)>0:
                asynchronous_pipeline_index_list.append([index, candidate_inputs])
    return asynchronous_pipeline_index_list

def judge_precompute_indices_condition(simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops):
    precompute_indices_index_list=[]
    len_eqs = len(eq_outputs_under_loops)
    for index in range(len_eqs):
        this_inputs = eq_inputs_under_loops[index]
        if len(simplified_eqs_under_loops[index])==1:
            this_output =  eq_outputs_under_loops[index][0]
            this_output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_output)
            input_remove_num_var_alpha_subscript_details=[]
            output_remove_num_var_alpha_subscript_details=[]
            if len(this_output_subscript_list)>0:
                output_subscript_details=generate_subscript_details(this_output_subscript_list)
                output_remove_num_var_alpha_subscript_details.extend([item for item in output_subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))==0 and not item.isdigit() and not item.isalpha()])
            for this_input in this_inputs:
                if this_input!=this_output:
                    this_input_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(this_input)
                    if len(this_input_subscript_list)>0:
                        input_subscript_details=generate_subscript_details(this_input_subscript_list)
                        input_remove_num_var_alpha_subscript_details.extend([item for item in input_subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))==0 and not item.isdigit() and not item.isalpha()])
            remove_num_var_alpha_subscript_details=list(set(output_remove_num_var_alpha_subscript_details+input_remove_num_var_alpha_subscript_details))
            if len(remove_num_var_alpha_subscript_details)>0:
                precompute_indices_index_list.append([index, remove_num_var_alpha_subscript_details])
    return precompute_indices_index_list

def judge_factorization_condition(simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    factorization_index_list = []
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        # print(f'this_eqs: {this_eqs}')
        this_outputs= simplified_eq_outputs_under_loops[index]
        variables = {var: symbols(var) for var in simplified_eq_inputs_under_loops[index]}
        for this_eq_index in range(len(this_eqs)):
            # print(f'this_eq:{this_eqs[this_eq_index]}')
            this_eq=this_eqs[this_eq_index]
            str_expr='='.join(this_eq.split('=')[1:])
            for this_output in this_outputs:
                if this_eq.startswith(this_output+'='+this_output+'+'):
                    str_expr = str_expr.replace(this_output+'+', '')
            if 'if_then_else' not in str_expr and 'erf' not in str_expr:
                # print(f'str_expr: {str_expr}')
                expr=sympify(str_expr, locals=variables, evaluate=False)
                str_original_expr=simplify_expr(str(expr))
                factor_expr = sqf(expr)#factor(expr)
                str_factor_expr=simplify_expr(str(factor_expr))
                # print(f'factor_expr:{str_factor_expr}, factor_expr:{factor_expr}')
                if str_factor_expr!= str_original_expr:
                    for this_output in this_outputs:
                        if this_eq.startswith(this_output+'='+this_output+'+'):
                            str_factor_expr=this_output+'+'+str_factor_expr
                        else:
                            str_factor_expr=str_factor_expr
                    factorization_index_list.append([index, this_eq_index, str_factor_expr])
    return factorization_index_list

def judge_expand_factorization_condition(simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    factorization_index_list = []
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        this_outputs= simplified_eq_outputs_under_loops[index]
        variables = {var: symbols(var) for var in simplified_eq_inputs_under_loops[index]}
        for this_eq_index in range(len(this_eqs)):
            # print(f'this_eq:{this_eqs[this_eq_index]}')
            this_eq=this_eqs[this_eq_index]
            str_expr='='.join(this_eq.split('=')[1:])
            for this_output in this_outputs:
                if this_eq.startswith(this_output+'='+this_output+'+'):
                    str_expr = str_expr.replace(this_output+'+', '')
            if 'if_then_else' not in str_expr and 'erf' not in str_expr:
                # print(f'str_expr: {str_expr}')
                expr=sympify(str_expr, locals=variables, evaluate=False)
                str_original_expr=simplify_expr(str(expr))
                expanded_expr = expand(expr)
                str_expanded_expr=simplify_expr(str(expanded_expr))
                # print(f'factor_expr:{str_factor_expr}, expr:{str_original_expr}')
                str_factor_expanded_expr=simplify_expr(str(factor(expanded_expr)))
                if str_expanded_expr != str_original_expr and str_factor_expanded_expr==str_original_expr:
                    for this_output in this_outputs:
                        if this_eq.startswith(this_output+'='+this_output+'+'):
                            str_expanded_expr=this_output+'+'+str_expanded_expr
                        else:
                            str_expanded_expr=str_expanded_expr
                    factorization_index_list.append([index, this_eq_index, str_expanded_expr])
    return factorization_index_list

def judge_cancellation_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    cancellation_index_list = []
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        this_outputs= simplified_eq_outputs_under_loops[index]
        variables = {var: symbols(var) for var in simplified_eq_inputs_under_loops[index]}
        for this_eq_index in range(len(this_eqs)):
            # print(f'this_eq:{this_eqs[this_eq_index]}')
            this_eq=this_eqs[this_eq_index]
            str_expr='='.join(this_eq.split('=')[1:])
            for this_output in this_outputs:
                if this_eq.startswith(this_output+'='+this_output+'+'):
                    str_expr = str_expr.replace(this_output+'+', '')
            if 'if_then_else' not in str_expr and '/' in str_expr and 'erf' not in str_expr:
                # print(f'str_expr: {str_expr}')
                expr=sympify(str_expr, locals=variables, evaluate=False)
                str_original_expr=simplify_expr(str(expr))
                cancelled_expr = cancel(expr)
                str_cancelled_expr=simplify_expr(str(cancelled_expr))
                factor_expr=factor(expr)
                str_factor_expr=simplify_expr(str(factor_expr))
                expanded_expr = expand(expr)
                str_expanded_expr=simplify_expr(str(expanded_expr))
                if str_cancelled_expr != str_original_expr and str_factor_expr!=str_cancelled_expr and str_expanded_expr!=str_cancelled_expr:
                    for this_output in this_outputs:
                        if this_eq.startswith(this_output+'='+this_output+'+'):
                            str_cancelled_expr=this_output+'+'+str_cancelled_expr
                        else:
                            str_cancelled_expr=str_cancelled_expr
                    cancellation_index_list.append([index, this_eq_index, str_cancelled_expr])
    return cancellation_index_list

def judge_apart_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    apart_index_list = []
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        this_outputs= simplified_eq_outputs_under_loops[index]
        variables = {var: symbols(var) for var in simplified_eq_inputs_under_loops[index]}
        for this_eq_index in range(len(this_eqs)):
            # print(f'this_eq:{this_eqs[this_eq_index]}')
            this_eq=this_eqs[this_eq_index]
            str_expr='='.join(this_eq.split('=')[1:])
            for this_output in this_outputs:
                if this_eq.startswith(this_output+'='+this_output+'+'):
                    str_expr = str_expr.replace(this_output+'+', '')
            temp_lower_str_expr = re.findall(r'\b[a-z]+\b',str_expr)
            lower_str_expr=[item for item in temp_lower_str_expr if item!='min' and item!='max' and item!='abs' and item!='sqrt' and item!='exp' and item!='log' and item!='sin' and item!='cos' and item!='tan' and item!='asin' and item!='acos' and item!='atan' and item!='erf']
            # print(f'lower_str_expr:{lower_str_expr}')
            if 'if_then_else' not in str_expr and 'erf' not in str_expr and 'abs' not in str_expr and 'max' not in str_expr and 'min' not in str_expr and 'exp' not in str_expr and 'log' not in str_expr and 'sin' not in str_expr and 'cos' not in str_expr and '/' in str_expr and len(variables)==1 and len(lower_str_expr)==0:
                # print(f'str_expr: {str_expr}')
                expr=sympify(str_expr, locals=variables, evaluate=False)
                str_original_expr=simplify_expr(str(expr))
                aparted_expr = apart(expr)
                str_aparted_expr=simplify_expr(str(aparted_expr))
                cancelled_expr = cancel(expr)
                str_cancelled_expr=simplify_expr(str(cancelled_expr))
                factor_expr=factor(expr)
                str_factor_expr=simplify_expr(str(factor_expr))
                expanded_expr = expand(expr)
                str_expanded_expr=simplify_expr(str(expanded_expr))
                if str_aparted_expr != str_original_expr and str_factor_expr!=str_aparted_expr and str_factor_expr!= str_cancelled_expr and str_expanded_expr!=str_aparted_expr:
                    for this_output in this_outputs:
                        if this_eq.startswith(this_output+'='+this_output+'+'):
                            str_aparted_expr=this_output+'+'+str_aparted_expr
                        else:
                            str_aparted_expr=str_aparted_expr
                    apart_index_list.append([index, this_eq_index, str_aparted_expr])
    return apart_index_list

def judge_together_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    together_index_list = []
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        this_outputs= simplified_eq_outputs_under_loops[index]
        variables = {var: symbols(var) for var in simplified_eq_inputs_under_loops[index]}
        for this_eq_index in range(len(this_eqs)):
            # print(f'this_eq:{this_eqs[this_eq_index]}')
            this_eq=this_eqs[this_eq_index]
            str_expr='='.join(this_eq.split('=')[1:])
            for this_output in this_outputs:
                if this_eq.startswith(this_output+'='+this_output+'+'):
                    str_expr = str_expr.replace(this_output+'+', '')
            # print(f'lower_str_expr:{lower_str_expr}')
            if 'if_then_else' not in str_expr and '/' in str_expr and 'erf' not in str_expr:
                # print(f'str_expr: {str_expr}')
                expr=sympify(str_expr, locals=variables, evaluate=False)
                str_original_expr=simplify_expr(str(expr))
                together_expr = together(expr)
                str_together_expr=simplify_expr(str(together_expr))
                cancelled_expr = cancel(expr)
                str_cancelled_expr=simplify_expr(str(cancelled_expr))
                factor_expr=factor(expr)
                str_factor_expr=simplify_expr(str(factor_expr))
                expanded_expr = expand(expr)
                str_expanded_expr=simplify_expr(str(expanded_expr))
                if str_together_expr != str_original_expr and str_factor_expr!=str_together_expr and str_together_expr!= str_cancelled_expr and str_expanded_expr!=str_together_expr:
                    for this_output in this_outputs:
                        if this_eq.startswith(this_output+'='+this_output+'+'):
                            str_together_expr=this_output+'+'+str_together_expr
                        else:
                            str_together_expr=str_together_expr
                    together_index_list.append([index, this_eq_index, str_together_expr])
    return together_index_list

def judge_trig_expand_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    trig_expand_index_list = []
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        this_outputs= simplified_eq_outputs_under_loops[index]
        variables = {var: symbols(var) for var in simplified_eq_inputs_under_loops[index]}
        for this_eq_index in range(len(this_eqs)):
            # print(f'this_eq:{this_eqs[this_eq_index]}')
            this_eq=this_eqs[this_eq_index]
            str_expr='='.join(this_eq.split('=')[1:])
            for this_output in this_outputs:
                if this_eq.startswith(this_output+'='+this_output+'+'):
                    str_expr = str_expr.replace(this_output+'+', '')
            # print(f'lower_str_expr:{lower_str_expr}')
            if 'if_then_else' not in str_expr and 'erf' not in str_expr:
                # print(f'str_expr: {str_expr}')
                expr=sympify(str_expr, locals=variables, evaluate=False)
                str_original_expr=simplify_expr(str(expr))
                trig_expand_expr = expand(expr, trig=True)
                str_trig_expand_expr=simplify_expr(str(trig_expand_expr))
                cancelled_expr = cancel(expr)
                str_cancelled_expr=simplify_expr(str(cancelled_expr))
                factor_expr=factor(expr)
                str_factor_expr=simplify_expr(str(factor_expr))
                expanded_expr = expand(expr)
                str_expanded_expr=simplify_expr(str(expanded_expr))
                if str_trig_expand_expr != str_original_expr and str_factor_expr!=str_trig_expand_expr and str_trig_expand_expr!= str_cancelled_expr and str_expanded_expr!=str_trig_expand_expr:
                    for this_output in this_outputs:
                        if this_eq.startswith(this_output+'='+this_output+'+'):
                            str_trig_expand_expr=this_output+'+'+str_trig_expand_expr
                        else:
                            str_trig_expand_expr=str_trig_expand_expr
                    trig_expand_index_list.append([index, this_eq_index, str_trig_expand_expr])
    return trig_expand_index_list

def judge_powsimp_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    powsimp_index_list = []
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        this_outputs= simplified_eq_outputs_under_loops[index]
        variables = {var: symbols(var) for var in simplified_eq_inputs_under_loops[index]}
        for this_eq_index in range(len(this_eqs)):
            # print(f'this_eq:{this_eqs[this_eq_index]}')
            this_eq=this_eqs[this_eq_index]
            str_expr='='.join(this_eq.split('=')[1:])
            for this_output in this_outputs:
                if this_eq.startswith(this_output+'='+this_output+'+'):
                    str_expr = str_expr.replace(this_output+'+', '')
            # print(f'lower_str_expr:{lower_str_expr}')
            if 'if_then_else' not in str_expr and 'erf' not in str_expr:
                # print(f'str_expr: {str_expr}')
                expr=sympify(str_expr, locals=variables, evaluate=False)
                str_original_expr=simplify_expr(str(expr))
                powsimp_expr = powsimp(expr)
                str_powsimp_expr=simplify_expr(str(powsimp_expr))
                if str_powsimp_expr != str_original_expr:
                    for this_output in this_outputs:
                        if this_eq.startswith(this_output+'='+this_output+'+'):
                            str_powsimp_expr=this_output+'+'+str_powsimp_expr
                        else:
                            str_powsimp_expr=str_powsimp_expr
                    powsimp_index_list.append([index, this_eq_index, str_powsimp_expr])
    return powsimp_index_list

def judge_expand_log_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    logcombine_index_list = []
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        this_outputs= simplified_eq_outputs_under_loops[index]
        variables = {var: symbols(var) for var in simplified_eq_inputs_under_loops[index]}
        for this_eq_index in range(len(this_eqs)):
            # print(f'this_eq:{this_eqs[this_eq_index]}')
            this_eq=this_eqs[this_eq_index]
            str_expr='='.join(this_eq.split('=')[1:])
            for this_output in this_outputs:
                if this_eq.startswith(this_output+'='+this_output+'+'):
                    str_expr = str_expr.replace(this_output+'+', '')
            # print(f'lower_str_expr:{lower_str_expr}')
            if 'if_then_else' not in str_expr and 'erf' not in str_expr:
                # print(f'str_expr: {str_expr}')
                expr=sympify(str_expr, locals=variables, evaluate=False)
                str_original_expr=simplify_expr(str(expr))
                logcombine_expr = logcombine(expr, force=True)
                str_logcombine_expr=simplify_expr(str(logcombine_expr))
                cancelled_expr = cancel(expr)
                str_cancelled_expr=simplify_expr(str(cancelled_expr))
                factor_expr=factor(expr)
                str_factor_expr=simplify_expr(str(factor_expr))
                expanded_expr = expand(expr)
                str_expanded_expr=simplify_expr(str(expanded_expr))
                if str_logcombine_expr != str_original_expr and str_factor_expr!=str_logcombine_expr and str_logcombine_expr!= str_cancelled_expr and str_expanded_expr!=str_logcombine_expr:
                    for this_output in this_outputs:
                        if this_eq.startswith(this_output+'='+this_output+'+'):
                            str_logcombine_expr=this_output+'+'+str_logcombine_expr
                        else:
                            str_logcombine_expr=str_logcombine_expr
                    logcombine_index_list.append([index, this_eq_index, str_logcombine_expr])
    return logcombine_index_list

def judge_collect_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    collect_index_list = []
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        this_outputs= simplified_eq_outputs_under_loops[index]
        variables = {var: symbols(var) for var in simplified_eq_inputs_under_loops[index]}
        for this_eq_index in range(len(this_eqs)):
            # print(f'this_eq:{this_eqs[this_eq_index]}')
            this_eq=this_eqs[this_eq_index]
            str_expr='='.join(this_eq.split('=')[1:])
            for this_output in this_outputs:
                if this_eq.startswith(this_output+'='+this_output+'+'):
                    str_expr = str_expr.replace(this_output+'+', '')
            # print(f'lower_str_expr:{lower_str_expr}')
            if 'if_then_else' not in str_expr and 'erf' not in str_expr:
                # print(f'str_expr: {str_expr}')
                expr=sympify(str_expr, locals=variables, evaluate=False)
                str_original_expr=simplify_expr(str(expr))
                cancelled_expr = cancel(expr)
                str_cancelled_expr=simplify_expr(str(cancelled_expr))
                factor_expr=factor(expr)
                str_factor_expr=simplify_expr(str(factor_expr))
                expanded_expr = expand(expr)
                str_expanded_expr=simplify_expr(str(expanded_expr))
                logcombine_expr = logcombine(expr)
                str_logcombine_expr=simplify_expr(str(logcombine_expr))
                for var in variables.values():
                    collect_expr = collect(expr, var)
                    str_collect_expr=simplify_expr(str(collect_expr))
                    if str_collect_expr != str_original_expr and str_collect_expr != str_logcombine_expr and str_collect_expr!=str_factor_expr and str_collect_expr!= str_cancelled_expr and str_collect_expr!=str_expanded_expr:
                        for this_output in this_outputs:
                            if this_eq.startswith(this_output+'='+this_output+'+'):
                                str_collect_expr=this_output+'+'+str_collect_expr
                            else:
                                str_collect_expr=str_collect_expr
                        collect_index_list.append([index, this_eq_index, str_collect_expr])
    return collect_index_list

def judge_partially_equivalent_then_correct_condition(loops,simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    #conv has the same shapes of the input and output
    #conv has the padding if_then_else
    #conv has the same stride, dilation and padding
    enable_partially_equivalent_then_correct=False
    # print('start judging')
    # print(f'simplified_eqs_under_loops:{simplified_eqs_under_loops}')
    if 'if_then_else' in simplified_eqs_under_loops[0][0] and 'if_then_else' in simplified_eqs_under_loops[2][0]:
        #in eq 0 and eq 2, we need to confirm the padding size, which is in subscript
        eq0_subscript_list, _ = find_subscripts_of_input_output_and_simplified_version(eq_inputs_under_loops[0][0])
        eq2_subscript_list, _ = find_subscripts_of_input_output_and_simplified_version(eq_inputs_under_loops[2][0])
        # print('if_then_else exist')
        if len(eq0_subscript_list)>0 and len(eq2_subscript_list)>0:
            # print('input has the subscript')
            if eq0_subscript_list[0]==eq2_subscript_list[0] and loops[0]==loops[2] and re.findall(r'[0-9]+',simplified_eqs_under_loops[0][0])==re.findall(r'[0-9]+',simplified_eqs_under_loops[2][0]):
                # print(f'conv has the same shapes of the input, padding size')
                if len(eq_inputs_under_loops[1])==len(eq_inputs_under_loops[3]):
                    # print(f'conv has the same number of inputs')
                    eq1_output_subscript_list, _ = find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[1][0])
                    eq3_output_subscript_list, _ = find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[3][0])
                    if eq1_output_subscript_list==eq3_output_subscript_list and loops[1]==loops[3]:
                        eq1_input_subscript_list,eq3_input_subscript_list,eq1_kernel_subscript_list,eq3_kernel_subscript_list=['eq1_input'], ['eq3_input'], ['eq1_kernel'], ['eq3_kernel']
                        # print(f'conv has the same output shapes')
                        for input_idx in range(len(eq_inputs_under_loops[1])):
                            # print(f'simplified_eq_outputs_under_loops[0][0]:{simplified_eq_outputs_under_loops[0][0]}, eq_inputs_under_loops[1][input_idx]:{eq_inputs_under_loops[1][input_idx]}')
                            if simplified_eq_outputs_under_loops[0][0] in eq_inputs_under_loops[1][input_idx]:
                                eq1_input_subscript_list,_ = find_subscripts_of_input_output_and_simplified_version(eq_inputs_under_loops[1][input_idx])
                            if simplified_eq_outputs_under_loops[2][0] in eq_inputs_under_loops[3][input_idx]:
                                eq3_input_subscript_list,_ = find_subscripts_of_input_output_and_simplified_version(eq_inputs_under_loops[3][input_idx])
                            if simplified_eq_outputs_under_loops[0][0] not in eq_inputs_under_loops[1][input_idx] and eq_outputs_under_loops[1][0]!=eq_inputs_under_loops[1][input_idx]:
                                eq1_kernel_subscript_list,_ = find_subscripts_of_input_output_and_simplified_version(eq_inputs_under_loops[1][input_idx])
                            if simplified_eq_outputs_under_loops[2][0] not in eq_inputs_under_loops[3][input_idx] and eq_outputs_under_loops[3][0]!=eq_inputs_under_loops[3][input_idx]:
                                eq3_kernel_subscript_list,_ = find_subscripts_of_input_output_and_simplified_version(eq_inputs_under_loops[3][input_idx])
                        if eq1_input_subscript_list==eq3_input_subscript_list and eq1_kernel_subscript_list==eq3_kernel_subscript_list:
                            # print(f'conv has the same stride, dilation and padding')
                            enable_partially_equivalent_then_correct=True
    return enable_partially_equivalent_then_correct

def judge_normal_loop_max_to_prefix_max_condition(simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops):
    normal_loop_max_to_prefix_max_index_list=[]
    len_eqs = len(simplified_eqs_under_loops)
    for index in range(len_eqs):
        this_eqs=simplified_eqs_under_loops[index]
        this_output=simplified_eq_outputs_under_loops[index]
        if len(this_eqs)==1 and len(this_output)==1:
            if this_output[0] in simplified_eq_inputs_under_loops[index] and 'max' in this_eqs[0] and len(eq_inputs_under_loops[index])==2:
                this_input_list=[item for item in eq_inputs_under_loops[index] if this_output[0] not in item]
                this_output_list=[item for item in eq_inputs_under_loops[index] if this_output[0] in item]
                this_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(this_input_list[0])
                this_output_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(this_output_list[0])
                if len(this_input_subscript_list)>0 and len(this_output_subscript_list)>0:
                    input_subscript_details=generate_subscript_details(this_input_subscript_list)
                    output_subscript_details=generate_subscript_details(this_output_subscript_list)
                    if len(input_subscript_details)-len(output_subscript_details)==1:
                        normal_loop_max_to_prefix_max_index_list.append(index)
    return normal_loop_max_to_prefix_max_index_list

def judge_exponential_split_condition(loops, simplified_eqs_under_loops):
    exponential_split_index_list=[]
    len_eqs= len(simplified_eqs_under_loops)
    for op_index in range(len_eqs):
        if len(loops[op_index])>0:
            this_eqs=simplified_eqs_under_loops[op_index]
            for eq_index in range(len(this_eqs)):
                this_eq=this_eqs[eq_index]
                if 'exp' in this_eq:
                    eq_right_part=''.join(this_eq.split('=')[1:])
                    eq_left_part=this_eq.split('=')[0]
                    if eq_left_part not in re.findall(r'exp\(.*?\)',eq_right_part)[0]:
                        exponential_split_index_list.append([op_index, eq_index])
    return exponential_split_index_list

def judge_multiplicative_split_condition(loops, simplified_eqs_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops):
    multiplicative_split_index_list=[]
    len_eqs= len(simplified_eqs_under_loops)
    for op_index in range(len_eqs):
        if len(loops[op_index])>0:
            this_eqs=simplified_eqs_under_loops[op_index]
            for eq_index in range(len(this_eqs)):
                this_eq=this_eqs[eq_index]
                if 'min' not in this_eq and 'max' not in this_eq and 'sqrt' not in this_eq and 'abs' not in this_eq and 'exp' not in this_eq and 'log' not in this_eq and 'if_then_else' not in this_eq and 'erf' not in this_eq and '**' not in this_eq:
                    eq_right_part=''.join(this_eq.split('=')[1:])
                    eq_left_part=this_eq.split('=')[0]
                    inputs_in_eq=re.findall(r'[A-Za-z]+',eq_right_part)
                    for input_in_eq in inputs_in_eq:
                        if input_in_eq!=eq_left_part and inputs_in_eq.count(input_in_eq)==1 and input_in_eq in simplified_eq_inputs_under_loops[op_index]:
                            full_inputs=[full_item for full_item in eq_inputs_under_loops[op_index] if re.sub(r'\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', full_item)==input_in_eq]
                            if len(full_inputs)==1:
                                multiplicative_split_index_list.append([op_index, eq_index, input_in_eq])
    return multiplicative_split_index_list

def judge_normal_loop_summation_on_exp_to_prefix_summation_on_exp_condition(loops, simplified_eqs_under_loops):
    normal_loop_summation_on_exp_to_prefix_summation_on_exp_index_list=[]
    len_eqs = len(simplified_eqs_under_loops)
    has_max=False
    max_input_output=[0,0]
    max_op_index=-1
    max_eq_index=-1
    for op_index in range(len_eqs):
        if len(loops[op_index])>0:
            this_eqs=simplified_eqs_under_loops[op_index]
            for eq_index in range(len(this_eqs)):
                this_eq=this_eqs[eq_index]
                if 'max' in this_eq:
                    max_simplified_split_eq=re.findall(r'[A-Za-z]+|[^a-zA-Z]',this_eq)
                    if len(max_simplified_split_eq)==8 and max_simplified_split_eq[0]==max_simplified_split_eq[4] and max_simplified_split_eq[1]=='=' and max_simplified_split_eq[2]=='max' and max_simplified_split_eq[3]=='(' and max_simplified_split_eq[5]==',' and max_simplified_split_eq[7]==')':
                        has_max=True
                        max_input_output=[max_simplified_split_eq[6], max_simplified_split_eq[0]]
                        max_op_index=op_index
                        max_eq_index=eq_index
                if has_max and 'exp' in this_eq and '+' in this_eq and '/' not in this_eq and 'if_then_else' not in this_eq and 'erf' not in this_eq and 'log' not in this_eq and 'sqrt' not in this_eq and 'abs' not in this_eq and 'min' not in this_eq and 'max' not in this_eq:
                    eq_right_part=''.join(this_eq.split('=')[1:])
                    eq_left_part=this_eq.split('=')[0]
                    simplified_split_eq_right_part=re.findall(r'[A-Za-z]+|[^a-zA-Z]',eq_right_part)
                    if len(simplified_split_eq_right_part)==8 and simplified_split_eq_right_part[0]==eq_left_part and simplified_split_eq_right_part[1]=='+' and simplified_split_eq_right_part[2]=='exp'and simplified_split_eq_right_part[3]=='(' and simplified_split_eq_right_part[4]==max_input_output[0] and simplified_split_eq_right_part[5]=='-' and simplified_split_eq_right_part[6]==max_input_output[1] and simplified_split_eq_right_part[7]==')':
                        normal_loop_summation_on_exp_to_prefix_summation_on_exp_index_list.append([op_index, eq_index, max_op_index, max_eq_index, simplified_split_eq_right_part])
    return normal_loop_summation_on_exp_to_prefix_summation_on_exp_index_list

def judge_online_softmax_condition(exp_op_index, exp_eq_index, simplified_split_exp_eq_right_part, loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    online_softmax_index_list=[]
    len_eqs = len(simplified_eqs_under_loops)
    for op_index in range(exp_op_index, len_eqs):
        if len(loops[op_index])>0:
            this_eqs=simplified_eqs_under_loops[op_index]
            for eq_index in range(len(this_eqs)):
                this_eq=this_eqs[eq_index]
                if 'exp' in this_eq and '/' in this_eq and 'if_then_else' not in this_eq and 'erf' not in this_eq and 'log' not in this_eq and 'sqrt' not in this_eq and 'abs' not in this_eq and 'min' not in this_eq and 'max' not in this_eq:
                    eq_left_part=this_eq.split('=')[0]
                    eq_right_part=''.join(this_eq.split('=')[1:])
                    simplified_split_eq_right_part=re.findall(r'[A-Za-z]+|[^a-zA-Z]',eq_right_part)
                    if len(simplified_split_eq_right_part)==8 and simplified_split_eq_right_part[0]=='exp' and simplified_split_eq_right_part[1]=='(' and simplified_split_eq_right_part[2]==simplified_split_exp_eq_right_part[4] and simplified_split_eq_right_part[3]=='-' and simplified_split_eq_right_part[4]==simplified_split_exp_eq_right_part[6] and simplified_split_eq_right_part[5]==')' and simplified_split_eq_right_part[6]=='/'  and simplified_split_eq_right_part[7]==simplified_split_exp_eq_right_part[0]:
                        online_softmax_index_list.append([op_index, eq_index, simplified_split_eq_right_part, eq_left_part])
    return online_softmax_index_list

def judge_flashattention_wo_tiling_condition(div_op_index, div_output, loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    flashattention_wo_tiling_list=[]
    len_eqs = len(simplified_eqs_under_loops)
    for op_index in range(div_op_index, len_eqs):
        if len(loops[op_index])>0:
            this_eqs=simplified_eqs_under_loops[op_index]
            for eq_index in range(len(this_eqs)):
                this_eq=this_eqs[eq_index]
                if '+' in this_eq and '*' in this_eq and 'exp' not in this_eq and '/' not in this_eq and 'if_then_else' not in this_eq and 'erf' not in this_eq and 'log' not in this_eq and 'sqrt' not in this_eq and 'abs' not in this_eq and 'min' not in this_eq and 'max' not in this_eq:
                    eq_left_part=this_eq.split('=')[0]
                    eq_right_part=''.join(this_eq.split('=')[1:])
                    simplified_split_eq_right_part=re.findall(r'[A-Za-z]+|[^a-zA-Z]',eq_right_part)
                    if len(simplified_split_eq_right_part)==5 and simplified_split_eq_right_part[0]==eq_left_part and simplified_split_eq_right_part[1]=='+' and (simplified_split_eq_right_part[2]==div_output or simplified_split_eq_right_part[4]==div_output) and simplified_split_eq_right_part[3]=='*':
                        flashattention_wo_tiling_list.append([op_index, eq_index, simplified_split_eq_right_part])
    return flashattention_wo_tiling_list

def apply_operator_fusion_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops,eq_outputs_under_loops,eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    fusion_index_list = judge_operator_fusion_condition(loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for fusion_index in fusion_index_list:
        # print(f'fusion_index:{fusion_index}')
        start_index, end_index = fusion_index
        fusion_loops=loops[start_index:end_index+1]
        distinct_loops = list(set(fusion_loops))
        fusion_eqs = equations_under_loops[start_index:end_index+1]
        temp_outputs_in_fusion_eqs = eq_outputs_under_loops[start_index:end_index+1]
        outputs_in_fusion_eqs=[iitem[:iitem.index('}')+1] for item in temp_outputs_in_fusion_eqs for iitem in item]
        later_IR=''.join(row_equations_under_loops[end_index+1:])
        # print(f'outputs_in_fusion_eqs:{outputs_in_fusion_eqs}')
        if len(distinct_loops)==1 or (len(distinct_loops)==2 and '' in distinct_loops):
            fusion_result=fusion_loops[0]+'['
            for eq in fusion_eqs:
                fusion_result+=';'.join(eq)+';'
            fusion_result+='];'
            # print(f'before fusion_result: {fusion_result}')
            for output_item in outputs_in_fusion_eqs:
                if output_item not in later_IR:
                    location_list=['l','s']
                    random_location=random.choice(location_list)
                    new_output_item=output_item.replace('g}',random_location+'}')
                    fusion_result=fusion_result.replace(output_item, new_output_item)
            # print(f'after fusion_result: {fusion_result}')
            transformed_IR = ''.join(row_equations_under_loops[:start_index])+ fusion_result + later_IR
            has_transformation=True
            transformed_IR_list.append(transformed_IR)
            original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation
        #TODO: no this case in our current IR
        # else:
        #     values_list, keys_list, _, len_list=split_loops_into_value_and_index(distinct_loops)
        #     if len(list(map(list, set(map(tuple,values_list)))))==1 and len(list(set(len_list)))==1:
        #         print(f'values_list: {values_list}, keys_list: {keys_list}, len_list: {len_list}')
                
def apply_operator_fission_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops,eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    fission_index_list=judge_operator_fission_condition(loops, equations_under_loops, simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for fission_index in fission_index_list:
        fission_loops=loops[fission_index]
        fission_eqs = equations_under_loops[fission_index]
        transformed_IR=''
        for eq in fission_eqs:
            transformed_IR+=fission_loops+'['+eq+';];'
        transformed_IR= ''.join(row_equations_under_loops[:fission_index]) + transformed_IR + ''.join(row_equations_under_loops[fission_index+1:])
        transformed_IR_list.append(transformed_IR)
        original_IR_list.append(IR)
        has_transformation=True
    return original_IR_list, transformed_IR_list, has_transformation

def apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops,eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    compute_inline_index_list=judge_compute_inline_condition(loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, row_equations_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for compute_inline_index in compute_inline_index_list:
        # print(f'compute_inline_index: {compute_inline_index}')
        start_index, end_index = compute_inline_index
        compute_inline_loops= loops[start_index] if loops[start_index] != '' else loops[end_index]
        previous_output = eq_outputs_under_loops[start_index][0]
        previous_eq = '='.join(equations_under_loops[start_index][0].split('=')[1:])
        this_eqs = equations_under_loops[end_index]
        transformed_eqs = []
        has_transformation = True
        for this_eq in this_eqs:
            if re.sub(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}', '', previous_eq)=='' or re.sub(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}', '', previous_eq)=='' or re.sub(r'if_then_else\(.*?\)', '', previous_eq)=='' or re.sub(r'min\(.*?\)', '', previous_eq)=='' or re.sub(r'max\(.*?\)', '', previous_eq)=='' or re.sub(r'sqrt\(.*?\)', '', previous_eq)=='' or re.sub(r'abs\(.*?\)', '', previous_eq)=='' or re.sub(r'exp\(.*?\)', '', previous_eq)=='' or re.sub(r'log\(.*?\)', '', previous_eq)=='' or re.sub(r'\d','', previous_eq)=='' or re.sub(r'\d','', previous_eq)=='.' or re.sub(r'\d','', previous_eq)=='-' or re.sub(r'\d','', previous_eq)=='-.':
                transformed_eq = this_eq.replace(previous_output, previous_eq)
            else:
                transformed_eq = this_eq.replace(previous_output, '('+previous_eq+')')
            transformed_eqs.append(transformed_eq)
        transformed_part=compute_inline_loops + '[' + ';'.join(transformed_eqs) + ';];'
        transformed_IR = ''.join(row_equations_under_loops[:start_index]) + transformed_part+ ''.join(row_equations_under_loops[end_index+1:])
        transformed_IR_list.append(transformed_IR)
        original_IR_list.append(IR)
        # print(f'transformed part: {transformed_part}')
        # print(f'transformed_IR: {transformed_IR}')
    return original_IR_list, transformed_IR_list, has_transformation

def apply_expression_splitting_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    # print(f'simplified_eqs_under_loops:{simplified_eqs_under_loops}')
    expression_splitting_index_list=judge_expression_splitting_condition(simplified_eqs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    # print(f'expression_splitting_index_list: {expression_splitting_index_list}')
    for expression_splitting_index in expression_splitting_index_list:
        has_transformation = True
        index, eq_index, simplified_expr = expression_splitting_index
        print(f'simplified_expr:{simplified_expr}')
        # print(f'index: {index}, eq_index: {eq_index}, simplified_expr: {simplified_expr}, equations_under_loops:{equations_under_loops[index][eq_index]}')
        expr, scripts=transform_from_original_simplified_expr_to_original_expr(simplified_expr, simplified_eq_inputs_under_loops[index], eq_inputs_under_loops[index])
        print(f'expr: {expr}, scripts: {scripts}')
        output_superscript = re.findall(r'\^\{.*?\}', eq_outputs_under_loops[index][0])[0]
        intermediate_names, name_start_idx = generate_names(1, name_start_idx)
        # print(f'output_superscript:{output_superscript}')
        intermediate_scripts, has_script=generate_expression_splitting_intermediate_scripts(scripts, output_superscript, expr)
        intermediate_scripts=intermediate_scripts.replace(' ','')
        eq=equations_under_loops[index][eq_index]
        intermediate_var=intermediate_names[0]+intermediate_scripts
        eq_loop= loops[index]
        if has_script:
            intermediate_subscript_related=intermediate_scripts[intermediate_scripts.index('_')+2:-1]
            intermediate_subscript=intermediate_subscript_related.split(',')
            intermediate_subscript_details=list(set(re.findall(rf'[a-z]+',intermediate_subscript_related)))
            value_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([eq_loop])
            key_value_dict= dict(zip(keys_list[0], value_list[0]))
            # print(f'key_value_dict: {key_value_dict}')
            intermediate_value_list=[eval(sub, {}, key_value_dict) for sub in intermediate_subscript]
            # print(f'intermediate_value_list: {intermediate_value_list}')
            new_value_list = [value_list[0][keys_list[0].index(sub_item)] for sub_item in intermediate_subscript_details]
            new_loop_type_list = [loop_type_list[0][keys_list[0].index(sub_item)] for sub_item in intermediate_subscript_details]
            if 'tx' not in intermediate_subscript_related:
                split_intermediate_subscript_related=re.findall(r'[a-zA-Z]+|[^a-zA-Z]',intermediate_subscript_related)
                original_key=intermediate_subscript_details[0]
                new_intermediate_subscript_details=['tx']+intermediate_subscript_details[1:]
                new_split_intermediate_subscript_related=[]
                for item in split_intermediate_subscript_related:
                    if item==original_key:
                        new_split_intermediate_subscript_related.append('tx')
                    else:
                        new_split_intermediate_subscript_related.append(item)
                new_intermediate_subscript_related='_{'+''.join(new_split_intermediate_subscript_related)+'}'
                new_intermediate_var=intermediate_var.replace('_{'+intermediate_subscript_related+'}', new_intermediate_subscript_related)
                # new_expr=expr.replace('_{'+intermediate_subscript_related+'}', new_intermediate_subscript_related)
                split_expr= re.findall(r'[A-Za-z]+|[^a-zA-Z]', expr)
                new_split_expr=[]
                for item in split_expr:
                    if item==original_key:
                        new_split_expr.append('tx')
                    else:
                        new_split_expr.append(item)
                new_expr=''.join(new_split_expr)
                new_loop_type_list=['B']+new_loop_type_list[1:]
            else:
                new_intermediate_var=intermediate_var
                new_intermediate_subscript_details=intermediate_subscript_details
                new_expr=expr
            new_loop=curate_loops(new_value_list, new_intermediate_subscript_details, new_loop_type_list)
            if math.prod(intermediate_value_list)<2147483647:
                transform_part = new_loop+'['+new_intermediate_var+'='+ new_expr + ';];'
            else:
                has_transformation=False
        else:
            transform_part = 'B^{1}_{tx=0}['+intermediate_var+'='+ expr + ';];'
        if has_transformation:
            transform_part += eq_loop+'['+eq.replace(expr, intermediate_var) + ';];'
            transformed_IR = ''.join(row_equations_under_loops[:index]) + transform_part + ''.join(row_equations_under_loops[index+1:])
            # print(f'transformed_IR: {transformed_IR}')
            transformed_IR_list.append(transformed_IR)
            original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_tensor_concat_to_fuse_operators_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    tensor_concat_to_fuse_operators_index_list=judge_tensor_concat_to_fuse_operators_condition(loops,simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    # print(f'tensor_concat_to_fuse_operators_index_list: {tensor_concat_to_fuse_operators_index_list}')
    for index in range(len(tensor_concat_to_fuse_operators_index_list)):
        has_transformation = True
        transform_IR=''
        op_index, last_eq_idx, _ = tensor_concat_to_fuse_operators_index_list[index]
        last_values_list, last_keys_list, last_loop_type_list, _ = split_loops_into_value_and_index([loops[op_index-1]])
        this_values_list, _, _, _ = split_loops_into_value_and_index([loops[op_index]])
        last_inputs = eq_inputs_under_loops[op_index-1]
        last_outputs = eq_outputs_under_loops[op_index-1]
        last_right_simplified_eq='='.join(simplified_eqs_under_loops[op_index-1][last_eq_idx].split('=')[1:])
        updated_last_inputs=[item for item in last_inputs if item not in last_outputs]
        this_inputs = eq_inputs_under_loops[op_index]
        this_outputs = eq_outputs_under_loops[op_index]
        this_right_simplified_eq='='.join(simplified_eqs_under_loops[op_index][last_eq_idx].split('=')[1:])
        # print(f'last_right_simplified_eq:{last_right_simplified_eq}, this_right_simplified_eq:{this_right_simplified_eq}')
        updated_this_inputs = [item for item in this_inputs if item not in this_outputs]
        output_superscript = re.findall(r'\^\{.*?\}', eq_outputs_under_loops[index][0])[0]
        output_subscript_list,_= find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[index][0])
        output_subscript = output_subscript_list[0]
        num_cancat_inputs, last_inputs_name_list, this_inputs_name_list, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, new_loop_notation, diff_last_keys, diff_value_index = find_different_vars_scripts_and_curate_loops_for_tensor_concat_fusion(last_values_list[0], last_keys_list[0], this_values_list[0], last_loop_type_list[0], updated_last_inputs, updated_this_inputs, last_right_simplified_eq, this_right_simplified_eq, output_subscript)
        # print(f'num_cancat_inputs:{num_cancat_inputs}, last_inputs_name_list:{last_inputs_name_list}, this_inputs_name_list:{this_inputs_name_list}, diff_last_keys:{diff_last_keys}')
        intermediate_names, name_start_idx = generate_names(num_cancat_inputs+1, name_start_idx)
        output_name=intermediate_names[-1]
        full_output_subscript=re.findall(r'[a-zA-Z]+|[^a-zA-Z]', output_subscript)
        output_resubscript = rewrite_subscript_for_cancat(full_output_subscript, diff_last_keys, diff_value_index, last_values_list[0])
        # print(f'output_name:{output_name}')
        # concat and split
        last_concat_IR = loops[op_index-1]+'['
        this_concat_IR = loops[op_index]+'['
        for i in range(num_cancat_inputs-1):
            last_concat_IR += intermediate_names[i]+inputs_superscripts_list[i]+inputs_subscripts_list[i]+'='+last_inputs_name_list[i]+';'
            this_concat_IR += intermediate_names[i]+inputs_superscripts_list[i]+resubscript_list[i]+'='+this_inputs_name_list[i]+';'
        last_concat_IR += intermediate_names[num_cancat_inputs-1]+inputs_superscripts_list[num_cancat_inputs-1]+inputs_subscripts_list[num_cancat_inputs-1]+'='+last_inputs_name_list[num_cancat_inputs-1]+';];'
        this_concat_IR += intermediate_names[num_cancat_inputs-1]+inputs_superscripts_list[num_cancat_inputs-1]+resubscript_list[num_cancat_inputs-1]+'='+this_inputs_name_list[num_cancat_inputs-1]+';];'
        last_split_IR = loops[op_index-1]+'['+eq_outputs_under_loops[op_index-1][0]+'='+output_name+output_superscript+ output_subscript+';];'
        this_split_IR = loops[op_index]+'['+ eq_outputs_under_loops[op_index][0]+'='+output_name+output_superscript+ output_resubscript+';];'
        #fusion
        expr=transfrom_from_original_simpified_expr_to_modified_expr(last_right_simplified_eq,simplified_eq_inputs_under_loops[op_index-1], simplified_eq_outputs_under_loops[op_index-1], eq_inputs_under_loops[op_index-1], output_name,output_subscript,output_superscript, last_inputs_name_list, intermediate_names,inputs_superscripts_list,inputs_subscripts_list)
        fuse_IR=new_loop_notation+'['+ output_name+output_superscript+ output_subscript+'='+expr + ';];'
        transform_part= last_concat_IR + this_concat_IR + fuse_IR + last_split_IR + this_split_IR
        transform_IR = ''.join(row_equations_under_loops[:op_index-1]) + transform_part + ''.join(row_equations_under_loops[op_index+1:])
        # print(f'transform_IR: {transform_IR}')
        transformed_IR_list.append(transform_IR)
        original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_tensor_split_to_decouple_operators_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    tensor_split_to_decouple_operators_index_list=judge_tensor_split_to_decouple_operators_condition(loops, simplified_eq_outputs_under_loops,simplified_eq_inputs_under_loops, eq_outputs_under_loops,eq_inputs_under_loops, simplified_eqs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(tensor_split_to_decouple_operators_index_list)):
        op_index, eq_idx, split_axis, split_reduce_aix=tensor_split_to_decouple_operators_index_list[index]
        # print(f'op_index: {op_index}, eq_idx: {eq_idx}, split_axis:{split_axis}, split_reduce_aix:{split_reduce_aix}, simplified_eqs_under_loops:{simplified_eqs_under_loops[op_index]}')
        # print(f'eq: {equations_under_loops[op_index][eq_idx]}')
        this_loop = loops[op_index]
        output_superscript = re.findall(r'\^\{.*?\}', eq_outputs_under_loops[op_index][0])[0]
        output_subscript_list,_= find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[op_index][0])
        output_subscript = output_subscript_list[0] if len(output_subscript_list)>0 else ''
        inputs = eq_inputs_under_loops[op_index]
        this_outputs = eq_outputs_under_loops[op_index]
        this_right_simplified_eq='='.join(simplified_eqs_under_loops[op_index][eq_idx].split('=')[1:])
        updated_this_inputs = [item for item in inputs if item not in this_outputs]
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        #for split axis
        simplified_eq=simplified_eqs_under_loops[op_index][eq_idx]
        lower_case= re.findall(r'[a-z]+', simplified_eq.replace('if_then_else', '').replace('erf', '').replace('log', '').replace('sqrt', '').replace('abs', '').replace('exp', '').replace('min', '').replace('max', ''))
        if len(split_axis)>0  and len(lower_case)==0:
            s_axis=random.choice(split_axis)
            num_cancat_inputs, inputs_name_list, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, loop_notation1, loop_notation2,first_two_loop_notation1,first_two_loop_notation2, new_values_list1=find_different_vars_scripts_and_curate_loops_for_tensor_split_decouple(s_axis, values_list[0], keys_list[0], loop_type_list[0], updated_this_inputs, this_right_simplified_eq)
            # print(f'this_right_simplified_eq:{this_right_simplified_eq}, inputs_name_list:{inputs_name_list}\nresubscript_list:{resubscript_list}')
            if num_cancat_inputs>0:
                intermediate_names, name_start_idx = generate_names(num_cancat_inputs*2+2, name_start_idx)
                output_name=intermediate_names[-2:]
                full_output_subscript=re.findall(r'[a-zA-Z]+|[^a-zA-Z]', output_subscript)
                output_resubscript = [output_subscript, rewrite_subscript_for_cancat(full_output_subscript, [keys_list[0][s_axis]], [s_axis], new_values_list1)]
                #split and concat
                split_IR1=first_two_loop_notation1+'['
                split_IR2=first_two_loop_notation2+'['
                for i in range(num_cancat_inputs-1):
                    split_IR1 += intermediate_names[i*2]+inputs_superscripts_list[i]+inputs_subscripts_list[i]+'='+inputs_name_list[i]+';'
                    split_IR2 += intermediate_names[i*2+1]+inputs_superscripts_list[i]+inputs_subscripts_list[i]+'='+re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','',inputs_name_list[i]).replace('}}','}')+resubscript_list[i]+';'
                split_IR1 += intermediate_names[(num_cancat_inputs-1)*2]+inputs_superscripts_list[num_cancat_inputs-1]+inputs_subscripts_list[num_cancat_inputs-1]+'='+inputs_name_list[num_cancat_inputs-1]+';];'
                split_IR2 += intermediate_names[(num_cancat_inputs-1)*2+1]+inputs_superscripts_list[num_cancat_inputs-1]+inputs_subscripts_list[num_cancat_inputs-1]+'='+re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','',inputs_name_list[num_cancat_inputs-1]).replace('}}','}')+resubscript_list[num_cancat_inputs-1]+';];'
                concat_IR1=loop_notation1+'['+eq_outputs_under_loops[op_index][0]+'='+output_name[0]+output_superscript+output_resubscript[0]+';];'
                concat_IR2=loop_notation2+'['+eq_outputs_under_loops[op_index][0].replace(output_resubscript[0],output_resubscript[1])+'='+output_name[1]+output_superscript+output_resubscript[0]+';];'
                #decouple the equation
                expr1=transfrom_from_original_simpified_expr_to_modified_expr(this_right_simplified_eq,simplified_eq_inputs_under_loops[op_index], simplified_eq_outputs_under_loops[op_index],eq_inputs_under_loops[op_index],output_name[0],output_resubscript[0],output_superscript, inputs_name_list, intermediate_names[::2], inputs_superscripts_list, inputs_subscripts_list)
                expr2=transfrom_from_original_simpified_expr_to_modified_expr(this_right_simplified_eq,simplified_eq_inputs_under_loops[op_index], simplified_eq_outputs_under_loops[op_index],eq_inputs_under_loops[op_index],output_name[1],output_resubscript[0],output_superscript, inputs_name_list, intermediate_names[1::2], inputs_superscripts_list, inputs_subscripts_list)
                decouple_IR1=loop_notation1+'['+output_name[0]+output_superscript+output_subscript+'='+expr1+';];'
                decouple_IR2=loop_notation2+'['+output_name[1]+output_superscript+output_subscript+'='+expr2+';];'
                transformed_part = split_IR1 + split_IR2 + decouple_IR1 + decouple_IR2 + concat_IR1 + concat_IR2 
                transform_IR= ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                # print(f'transform_IR: {transformed_part}')
                value_list1, _, _, _ = split_loops_into_value_and_index([loop_notation1])
                value_list2, _, _, _ = split_loops_into_value_and_index([loop_notation2])
                if math.prod(value_list1[0])<=2147483648 and math.prod(value_list2[0])<=2147483648:
                    has_transformation = True
                    transformed_IR_list.append(transform_IR)
                    original_IR_list.append(IR)
        #for reduce axis
        simplified_output=simplified_eq_outputs_under_loops[op_index][0]
        simplified_input=simplified_eq_inputs_under_loops[op_index]
        # print(f'simplified_output:{simplified_output},simplified_input:{simplified_input}')
        if (len(lower_case)==0) and (len(split_reduce_aix)>0) and (simplified_output+'+' in simplified_eq or simplified_output+'*' in simplified_eq or 'min('+simplified_output in simplified_eq or 'max('+simplified_output in simplified_eq):
            r_axis=random.choice(split_reduce_aix)            
            num_cancat_inputs, inputs_name_list, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, loop_notation1, loop_notation2, first_two_loop_notation1,first_two_loop_notation2,new_values_list1=find_different_vars_scripts_and_curate_loops_for_tensor_split_decouple(r_axis, values_list[0], keys_list[0], loop_type_list[0], updated_this_inputs, this_right_simplified_eq)
            if num_cancat_inputs>0:
                new_inputs_subscripts_list=inputs_subscripts_list+[output_subscript]
                new_inputs_superscripts_list=inputs_superscripts_list+[output_superscript]
                intermediate_names, name_start_idx = generate_names(num_cancat_inputs*2+2, name_start_idx)
                output_name=intermediate_names[-2:]
                concat_values_list=values_list[0][:r_axis]+values_list[0][r_axis+1:]
                concat_keys_list=keys_list[0][:r_axis]+keys_list[0][r_axis+1:]
                concat_loop_type_list=loop_type_list[0][:r_axis]+loop_type_list[0][r_axis+1:]
                output_subscript_details = re.findall(r'\b[a-zA-Z]+\b', output_subscript)
                if len(output_subscript_details)<len(concat_values_list)-1 and len(concat_values_list)>0:
                    output_subscript='_{'+','.join(concat_keys_list)+'}'
                full_output_subscript=re.findall(r'[a-zA-Z]+|[^a-zA-Z]', output_subscript)
                output_resubscript = [output_subscript, rewrite_subscript_for_cancat(full_output_subscript, [keys_list[0][r_axis]], [r_axis], new_values_list1)]
                #split and concat
                split_IR1=first_two_loop_notation1+'['
                split_IR2=first_two_loop_notation2+'['
                for i in range(num_cancat_inputs-1):
                    split_IR1 += intermediate_names[i*2]+inputs_superscripts_list[i]+inputs_subscripts_list[i]+'='+inputs_name_list[i]+';'
                    split_IR2 += intermediate_names[i*2+1]+inputs_superscripts_list[i]+inputs_subscripts_list[i]+'='+re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','',inputs_name_list[i]).replace('}}','}')+resubscript_list[i]+';'
                split_IR1 += intermediate_names[(num_cancat_inputs-1)*2]+inputs_superscripts_list[num_cancat_inputs-1]+inputs_subscripts_list[num_cancat_inputs-1]+'='+inputs_name_list[num_cancat_inputs-1]+';];'
                split_IR2 += intermediate_names[(num_cancat_inputs-1)*2+1]+inputs_superscripts_list[num_cancat_inputs-1]+inputs_subscripts_list[num_cancat_inputs-1]+'='+re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','',inputs_name_list[num_cancat_inputs-1]).replace('}}','}')+resubscript_list[num_cancat_inputs-1]+';];'
                if 'tx' not in concat_keys_list:
                    concat_values_list=[1]+concat_values_list
                    concat_keys_list=['tx']+concat_keys_list
                    concat_loop_type_list=['B']+concat_loop_type_list
                concat_loop_notation=curate_loops(concat_values_list, concat_keys_list, concat_loop_type_list)
                input_same_as_output=[item for item in eq_inputs_under_loops[op_index] if simplified_output+'^' in item]
                if simplified_output not in simplified_input:
                    if simplified_output+'+' in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'='+output_name[0]+output_superscript+output_resubscript[0]+'+'+output_name[1]+output_superscript+output_resubscript[1]+';];'
                    elif simplified_output+'*' in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'='+output_name[0]+output_superscript+output_resubscript[0]+'*'+output_name[1]+output_superscript+output_resubscript[1]+';];'
                    elif 'min('+simplified_output in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'=min('+output_name[0]+output_superscript+output_resubscript[0]+','+output_name[1]+output_superscript+output_resubscript[1]+');];'
                    elif 'max('+simplified_output in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'=max('+output_name[0]+output_superscript+output_resubscript[0]+','+output_name[1]+output_superscript+output_resubscript[1]+');];'
                else:
                    if simplified_output+'+' in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'='+input_same_as_output[0]+'+('+output_name[0]+output_superscript+output_resubscript[0]+'+'+output_name[1]+output_superscript+output_resubscript[1]+');];'
                    elif simplified_output+'*' in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'='+input_same_as_output[0]+'*('+output_name[0]+output_superscript+output_resubscript[0]+'*'+output_name[1]+output_superscript+output_resubscript[1]+');];'
                    elif 'min('+simplified_output in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'=min('+input_same_as_output[0]+',min('+output_name[0]+output_superscript+output_resubscript[0]+','+output_name[1]+output_superscript+output_resubscript[1]+'));];'
                    elif 'max('+simplified_output in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'=max('+input_same_as_output[0]+',max('+output_name[0]+output_superscript+output_resubscript[0]+','+output_name[1]+output_superscript+output_resubscript[1]+'));];'
                #decouple the equation
                expr1=transfrom_from_original_simpified_expr_to_modified_expr('='.join(simplified_eq.split('=')[1:]),simplified_eq_inputs_under_loops[op_index],simplified_eq_outputs_under_loops[op_index], eq_inputs_under_loops[op_index],output_name[0],output_resubscript[0],output_superscript,  inputs_name_list, intermediate_names[::2], new_inputs_superscripts_list, new_inputs_subscripts_list)
                expr2=transfrom_from_original_simpified_expr_to_modified_expr('='.join(simplified_eq.split('=')[1:]),simplified_eq_inputs_under_loops[op_index],simplified_eq_outputs_under_loops[op_index], eq_inputs_under_loops[op_index],output_name[1],output_resubscript[1],output_superscript,  inputs_name_list, intermediate_names[1::2], new_inputs_superscripts_list, new_inputs_subscripts_list)
                decouple_IR1=loop_notation1+'['+output_name[0]+output_superscript+output_subscript+'='+expr1+';];'
                decouple_IR2=loop_notation2+'['+output_name[1]+output_superscript+output_subscript+'='+expr2+';];'
                transformed_part = split_IR1 + split_IR2 + decouple_IR1 + decouple_IR2 + concat_IR 
                transform_IR= ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                # print(f'transform_IR: {transformed_part}')
                if math.prod(concat_values_list)<=2147483648:
                    has_transformation = True
                    transformed_IR_list.append(transform_IR)
                    original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_common_subexpression_elimination_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    common_subexpression_elimination_index_mapping=judge_common_subexpression_elimination_condition(loops, simplified_eqs_under_loops, eq_outputs_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    subexpression_mapping={}
    for key, values in common_subexpression_elimination_index_mapping.items():
        op_index, eq_index = eval(key)
        # print(f'op_index: {op_index}, eq_index: {eq_index}, values: {values}')
        for value in values:
            if value not in subexpression_mapping.keys():
                has_transformation = True
                intermediate_names, name_start_idx = generate_names(1, name_start_idx)
                subscripts=find_all_subscripts_in_expr(value)
                # print(f'value:{value}, subscripts: {subscripts}')
                superscript=re.findall(r'\^\{.*?\}', value)[0] if len(re.findall(r'\^\{.*?\}', value))>0 else ''
                # print(f'1superscript: {superscript}')
                if superscript=='':
                    this_output= eq_outputs_under_loops[op_index][0]
                    superscript = re.findall(r'\^\{.*?\}', this_output)[0] if len(re.findall(r'\^\{.*?\}', this_output))>0 else ''
                # print(f'2superscript: {superscript}')
                # print
                this_loop = loops[op_index]
                # print(f'this_loop: {this_loop}')
                values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
                loop_notation, output_subscript=curate_loops_and_output_subscript_according_to_subscripts(keys_list[0], values_list[0], loop_type_list[0], subscripts)
                if loop_notation=='':
                    loop_notation = 'B^{1}_{tx=0}'
                intermediate_var=intermediate_names[0]+superscript+output_subscript
                subexpression_expr = loop_notation + '[' + intermediate_var + '=' + value + ';];'
                subexpression_mapping[value] = [intermediate_var, subexpression_expr]
                row_equations_under_loops[op_index]=subexpression_expr+ row_equations_under_loops[op_index].replace(value, intermediate_var)
            else:
                intermediate_var, subexpression_expr = subexpression_mapping[value]
                row_equations_under_loops[op_index]= row_equations_under_loops[op_index].replace(value, intermediate_var)
    transformed_IR = ''.join(row_equations_under_loops)
    if has_transformation:
        return [IR], [transformed_IR], has_transformation 
    else:
        return [IR], [], has_transformation

def apply_dead_code_elimination_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    dead_code_elimination_index_list=judge_dead_code_elimination_condition(simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    if len(dead_code_elimination_index_list)>0:
        has_transformation = True
        for index in dead_code_elimination_index_list:
            transformed_IR = ''.join(row_equations_under_loops[:index]) + ''.join(row_equations_under_loops[index+1:])
            transformed_IR_list.append(transformed_IR)
        return [IR], transformed_IR_list, has_transformation
    else:
        has_transformation = True
        transformed_IR_list = [IR]
        duplicate_index = random.randint(0, len(row_equations_under_loops)-1)
        insert_index = random.randint(duplicate_index, len(row_equations_under_loops)-1)
        duplicate_expression = row_equations_under_loops[duplicate_index]
        duplicate_output = eq_outputs_under_loops[duplicate_index][0]
        simplified_duplicate_output = simplified_eq_outputs_under_loops[duplicate_index][0]
        intermediate_names, name_start_idx = generate_names(1, name_start_idx)
        new_output = duplicate_output.replace(simplified_duplicate_output, intermediate_names[0])
        new_expression = duplicate_expression.replace(duplicate_output, new_output)
        original_IR = ''.join(row_equations_under_loops[:insert_index]) + new_expression + ''.join(row_equations_under_loops[insert_index:])
        # print(f'original_IR: {original_IR}')
        # print(f'transformed_IR_list: {transformed_IR_list}')
        return [original_IR], transformed_IR_list, has_transformation

def apply_expression_reorder_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    expression_reorder_index_list=judge_expression_reorder_condition(simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for op_index in expression_reorder_index_list:
        has_transformation = True
        transformed_IR= ''.join(row_equations_under_loops[:op_index])+row_equations_under_loops[op_index+1]+row_equations_under_loops[op_index]+''.join(row_equations_under_loops[op_index+2:])
        transformed_IR_list.append(transformed_IR)
        original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_loop_reorder_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    loop_reorder_index_list=judge_loop_reorder_condition(loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for op_index in loop_reorder_index_list:
        has_transformation = True
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([loops[op_index]])
        # bx_index = keys_list[0].index('bx')
        # tx_index = keys_list[0].index('tx')
        selected_index_list = list(range(len(keys_list[0])))#list(set(list(range(len(keys_list[0]))))-set([tx_index]))
        selected_index = random.sample(selected_index_list, 2)
        selected_index.sort()
        new_keys_list = keys_list[0][:selected_index[0]]+[keys_list[0][selected_index[1]]]+ keys_list[0][selected_index[0]+1:selected_index[1]] +[keys_list[0][selected_index[0]]]+ keys_list[0][selected_index[1]+1:]
        new_values_list = values_list[0][:selected_index[0]] +[values_list[0][selected_index[1]]]+ values_list[0][selected_index[0]+1:selected_index[1]] +[values_list[0][selected_index[0]]]+ values_list[0][selected_index[1]+1:]
        new_loop_type_list = loop_type_list[0][:selected_index[0]] +[loop_type_list[0][selected_index[1]]]+ loop_type_list[0][selected_index[0]+1:selected_index[1]] +[loop_type_list[0][selected_index[0]]]+ loop_type_list[0][selected_index[1]+1:]
        # if selected_index[0]== bx_index:
        #     new_keys_list = keys_list[0][:bx_index] + [keys_list[0][selected_index[1]]] + keys_list[0][tx_index+1:selected_index[1]]+[keys_list[0][bx_index], keys_list[0][tx_index]] + keys_list[0][selected_index[1]+1:]
        #     new_values_list = values_list[0][:bx_index] + [values_list[0][selected_index[1]]] + values_list[0][tx_index+1:selected_index[1]]+[values_list[0][bx_index], values_list[0][tx_index]] + values_list[0][selected_index[1]+1:]
        #     new_loop_type_list = loop_type_list[0][:bx_index] + [loop_type_list[0][selected_index[1]]] + loop_type_list[0][tx_index+1:selected_index[1]]+[loop_type_list[0][bx_index], loop_type_list[0][tx_index]] + loop_type_list[0][selected_index[1]+1:]
        # elif selected_index[1]== bx_index:
        #     new_keys_list = keys_list[0][:selected_index[0]] + [keys_list[0][bx_index], keys_list[0][tx_index]] + keys_list[0][selected_index[0]+1:selected_index[1]]+[keys_list[0][selected_index[0]]] + keys_list[0][selected_index[1]+1:]
        #     new_values_list = values_list[0][:selected_index[0]] + [values_list[0][bx_index], values_list[0][tx_index]] + values_list[0][selected_index[0]+1:selected_index[1]]+[values_list[0][selected_index[0]]] + values_list[0][selected_index[1]+1:]
        #     new_loop_type_list = loop_type_list[0][:selected_index[0]] + [loop_type_list[0][bx_index], loop_type_list[0][tx_index]] + loop_type_list[0][selected_index[0]+1:selected_index[1]]+[loop_type_list[0][selected_index[0]]] + loop_type_list[0][selected_index[1]+1:]
        # else:
        #     new_keys_list = keys_list[0][:selected_index[0]] + [keys_list[0][bx_index], keys_list[0][tx_index]] + keys_list[0][selected_index[0]:selected_index[1]] + keys_list[0][selected_index[1]+1:]
        #     new_values_list = values_list[0][:selected_index[0]] + [values_list[0][bx_index], values_list[0][tx_index]] + values_list[0][selected_index[0]:selected_index[1]] + values_list[0][selected_index[1]+1:]
        #     new_loop_type_list = loop_type_list[0][:selected_index[0]] + [loop_type_list[0][bx_index], loop_type_list[0][tx_index]] + loop_type_list[0][selected_index[0]:selected_index[1]] + loop_type_list[0][selected_index[1]+1:]
        new_loop_notation = curate_loops(new_values_list, new_keys_list, new_loop_type_list)
        new_raw_eq=new_loop_notation+'['+';'.join(equations_under_loops[op_index])+';];'
        transformed_IR = ''.join(row_equations_under_loops[:op_index]) + new_raw_eq + ''.join(row_equations_under_loops[op_index+1:])
        transformed_IR_list.append(transformed_IR)
        original_IR_list.append(IR)
        # print(f'row_equations_under_loops[:op_index]: {row_equations_under_loops[op_index]}, new_raw_eq:{new_raw_eq}, selected_index:{selected_index}')
    return original_IR_list, transformed_IR_list, has_transformation

# def apply_loop_tiling_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
#     loop_tiling_index_list=judge_loop_tiling_condition(loops)
#     has_transformation = False
#     transformed_IR_list=[]
#     for index in range(len(loop_tiling_index_list)):
#         has_transformation = True
#         op_index, loop_idx, this_value, next_value = loop_tiling_index_list[index]
#         # print(f'op_index: {op_index}, loop_idx: {loop_idx}')
#         values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([loops[op_index]])
#         this_eqs= ','.join(equations_under_loops[op_index])
#         bx_index = keys_list[0].index('bx')
#         intermediate_keys=generate_idx_names(2, len(keys_list[0])-2)
#         selected_this_value=random.choice(this_value)
#         selected_next_value=random.choice(next_value)
#         # print(f'selected_this_value: {selected_this_value}, selected_next_value: {selected_next_value}')
#         if loop_idx==bx_index-1:
#             #this loop: normal loop, next loop: binding loop bx* +tx
#             new_keys_list = keys_list[0][:loop_idx]+ intermediate_keys+ keys_list[0][loop_idx:]
#             new_values_list = values_list[0][:loop_idx]+ [selected_this_value[0], selected_next_value[0], selected_this_value[1], 1, selected_next_value[1]] + values_list[0][loop_idx+3:]
#             new_loop_type_list = loop_type_list[0][:loop_idx]+ ['L', 'L'] + loop_type_list[0][loop_idx:]
#             original_subscript=[keys_list[0][loop_idx], 'bx*'+str(values_list[0][bx_index+1])+'+tx']
#             new_subscript=[intermediate_keys[0]+'*'+str(selected_this_value[0])+'+'+keys_list[0][loop_idx], intermediate_keys[1]+'*'+str(selected_next_value[0])+'+bx*'+str(selected_next_value[1])+'+tx']
#         elif loop_idx==bx_index+1:
#             #this loop: binding loop bx* +tx, next loop: normal loop
#             new_keys_list = keys_list[0][:bx_index] + intermediate_keys + keys_list[0][bx_index:]
#             new_values_list = values_list[0][:bx_index] + [selected_this_value[0], selected_next_value[0], 1, selected_this_value[1], selected_next_value[1]] + values_list[0][bx_index+3:]
#             new_loop_type_list = loop_type_list[0][:bx_index] + ['L', 'L'] + loop_type_list[0][bx_index:]
#             original_subscript=['bx*'+str(values_list[0][bx_index+1])+'+tx', keys_list[0][loop_idx+1]]
#             new_subscript=[intermediate_keys[0]+'*'+str(selected_this_value[0])+'+bx*'+str(selected_this_value[1])+'+tx', intermediate_keys[1]+'*'+str(selected_next_value[0])+'+'+keys_list[0][loop_idx+1]]
#         else:
#             #this loop, next loop: normal loops
#             new_keys_list = keys_list[0][:loop_idx] + intermediate_keys + keys_list[0][loop_idx:]
#             new_values_list = values_list[0][:loop_idx] + [selected_this_value[0], selected_next_value[0], selected_this_value[1], selected_next_value[1]] + values_list[0][loop_idx+2:]
#             new_loop_type_list = loop_type_list[0][:loop_idx] + ['L', 'L'] + loop_type_list[0][loop_idx:]
#             original_subscript=[keys_list[0][loop_idx], keys_list[0][loop_idx+1]]
#             new_subscript=[intermediate_keys[0]+'*'+str(selected_this_value[0])+'+'+keys_list[0][loop_idx], intermediate_keys[1]+'*'+str(selected_next_value[0])+'+'+keys_list[0][loop_idx+1]]
#         new_loop_notation = curate_loops(new_values_list, new_keys_list, new_loop_type_list)
#         # print(f'this loop: {loops[op_index]},\n new_loop_notation: {new_loop_notation}')
#         # print(f'original_subscript: {original_subscript},\n new_subscript: {new_subscript}')
#         # print(f'this_eqs: {this_eqs}')
#         this_eqs=this_eqs.replace('{'+original_subscript[0], '{'+new_subscript[0]).replace(','+original_subscript[0], ','+new_subscript[0]).replace('+'+original_subscript[0], '+'+new_subscript[0]).replace('*'+original_subscript[0], '*'+new_subscript[0])
#         this_eqs=this_eqs.replace('{'+original_subscript[1], '{'+new_subscript[1]).replace(','+original_subscript[1], ','+new_subscript[1]).replace('+'+original_subscript[1], '+'+new_subscript[1]).replace('*'+original_subscript[1], '*'+new_subscript[1])
#         # print(f'modified_this_eqs: {this_eqs}')
#         transformed_part=new_loop_notation+'['+this_eqs+';];'
#         transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
#         transformed_IR_list.append(transformed_IR)
#         # print(f'transformed_part: {transformed_part}')
#     return [IR], transformed_IR_list, has_transformation

def apply_loop_tiling_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    loop_tiling_index_list=judge_loop_tiling_condition(loops,eq_outputs_under_loops,simplified_eqs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(loop_tiling_index_list)):
        has_transformation = True
        op_index, loop_idx, this_value, next_value = loop_tiling_index_list[index]
        # print(f'op_index: {op_index}, loop_idx: {loop_idx}')
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([loops[op_index]])
        # print(f'values_list: {values_list}, keys_list: {keys_list}, loop_type_list: {loop_type_list}')
        this_eqs= ';'.join(equations_under_loops[op_index])
        intermediate_keys=generate_idx_names(2, len(keys_list[0])+1)
        selected_this_value=random.choice(this_value)
        selected_next_value=random.choice(next_value)
        new_keys_list = keys_list[0][:loop_idx] + intermediate_keys + keys_list[0][loop_idx:]
        new_values_list = values_list[0][:loop_idx] + [selected_this_value[0], selected_next_value[0], selected_this_value[1], selected_next_value[1]] + values_list[0][loop_idx+2:]
        new_loop_type_list = loop_type_list[0][:loop_idx] + ['L', 'L'] + loop_type_list[0][loop_idx:]
        # print(f'new_keys_list:{new_keys_list},new_values_list:{new_values_list},new_loop_type_list:{new_loop_type_list}')
        original_subscript=[keys_list[0][loop_idx], keys_list[0][loop_idx+1]]
        new_subscript=[intermediate_keys[0]+'*'+str(values_list[0][loop_idx]//selected_this_value[0])+'+'+keys_list[0][loop_idx], intermediate_keys[1]+'*'+str(values_list[0][loop_idx+1]//selected_next_value[0])+'+'+keys_list[0][loop_idx+1]]
        new_loop_notation = curate_loops(new_values_list, new_keys_list, new_loop_type_list)
        # print(f'this loop: {loops[op_index]},\n new_loop_notation: {new_loop_notation}')
        # print(f'original_subscript: {original_subscript},\n new_subscript: {new_subscript}')
        # print(f'this_eqs: {this_eqs}')
        split_this_eqs_by_subscript = split_eq_by_subscript(this_eqs)
        # print(f'split_this_eqs_by_subscript: {split_this_eqs_by_subscript}')
        replaced_split_eqs=replace_subscript_in_split_eqs(split_this_eqs_by_subscript, original_subscript[0], new_subscript[0])
        replaced_split_eqs=replace_subscript_in_split_eqs(replaced_split_eqs, original_subscript[1], new_subscript[1])
        this_eqs=''.join(replaced_split_eqs)
        # print(f'modified_this_eqs: {this_eqs}')
        transformed_part=new_loop_notation+'['+this_eqs+';];'
        transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
        transformed_IR_list.append(transformed_IR)
        original_IR_list.append(IR)
        # print(f'transformed_part: {transformed_part}')
    return original_IR_list, transformed_IR_list, has_transformation

def apply_loop_split_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    loop_split_index_list=judge_loop_split_condition(loops,eq_outputs_under_loops,simplified_eqs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(loop_split_index_list)):
        has_transformation = True
        op_index, loop_idx, this_value = loop_split_index_list[index]
        # print(f'op_index: {op_index}, loop_idx: {loop_idx}')
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([loops[op_index]])
        this_eqs= ';'.join(equations_under_loops[op_index])
        intermediate_keys=generate_idx_names(1, len(keys_list[0])+1)
        selected_this_value=random.choice(this_value)
        # print(f'selected_this_value: {selected_this_value}')
        new_keys_list = keys_list[0][:loop_idx] + intermediate_keys + keys_list[0][loop_idx:]
        new_values_list = values_list[0][:loop_idx] + [selected_this_value[0], selected_this_value[1]] + values_list[0][loop_idx+1:]
        new_loop_type_list = loop_type_list[0][:loop_idx] + ['L'] + loop_type_list[0][loop_idx:]
        new_subscript=intermediate_keys[0]+'*'+str(values_list[0][loop_idx]//selected_this_value[0])+'+'+keys_list[0][loop_idx]
        original_subscript=keys_list[0][loop_idx]
        new_loop_notation = curate_loops(new_values_list, new_keys_list, new_loop_type_list)
        # print(f'this_eqs: {this_eqs}')
        split_this_eqs_by_subscript = split_eq_by_subscript(this_eqs)
        # print(f'split_this_eqs_by_subscript: {split_this_eqs_by_subscript}')
        replaced_split_eqs=replace_subscript_in_split_eqs(split_this_eqs_by_subscript, original_subscript, new_subscript)
        this_eqs=''.join(replaced_split_eqs)
        # print(f'modified_this_eqs: {this_eqs}')
        # print(f'modified_this_eqs: {this_eqs}')
        transformed_part=new_loop_notation+'['+this_eqs+';];'
        transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
        transformed_IR_list.append(transformed_IR)
        original_IR_list.append(IR)
        # print(f'transformed_part: {transformed_part}')
    return original_IR_list, transformed_IR_list, has_transformation

def apply_loop_fusion_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    original_IR, transformed_IR_list, has_transformation=apply_loop_split_to_IR(IR,input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    return transformed_IR_list, original_IR, has_transformation

def apply_loop_unrolling_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    loop_unrolling_index_list=judge_loop_unrolling_condition(loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(loop_unrolling_index_list)):
        has_transformation = True
        op_index= loop_unrolling_index_list[index]
        # print(f'loop_unrolling_index_list: {op_index}')
        this_loop = loops[op_index]
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        tx_index = keys_list[0].index('tx')
        selected_index_list = list(set(list(range(len(keys_list[0]))))-set([tx_index]))
        selected=False
        while not selected and len(selected_index_list)>0:
            selected_index = random.choice(selected_index_list)
            other_values=values_list[0][:selected_index]+values_list[0][selected_index+1:]
            if values_list[0][selected_index]<1024 and math.prod(other_values)<2147483648/8:
                selected=True
                break
            else:
                selected_index_list.remove(selected_index)
        if selected:
            new_loop_type_list = loop_type_list[0][:selected_index] + ['U'] + loop_type_list[0][selected_index+1:]
            new_loop_notation = curate_loops(values_list[0], keys_list[0], new_loop_type_list)
            new_raw_eq=new_loop_notation+'['+';'.join(equations_under_loops[op_index])+';];'
            # print(f'new_raw_eq: {new_raw_eq}, old_eq: {row_equations_under_loops[op_index]}')
            transformed_IR = ''.join(row_equations_under_loops[:op_index]) + new_raw_eq + ''.join(row_equations_under_loops[op_index+1:])
            transformed_IR_list.append(transformed_IR)
            original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_loop_parallelization_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    loop_parallelization_index_list=judge_loop_parallelization_condition(loops, eq_outputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(loop_parallelization_index_list)):
        index, parallel_axis = loop_parallelization_index_list[index]
        this_loop = loops[index]
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        tx_index = keys_list[0].index('tx')
        can_be_selected_axis=[axis for axis in parallel_axis if keys_list[0].index(axis)>tx_index]
        if len(can_be_selected_axis)>0:
            has_transformation = True
            selected_axis = random.choice(parallel_axis)
            index_selected_axis = keys_list[0].index(selected_axis)
            new_loop_type_list = loop_type_list[0][:index_selected_axis] + ['P'] + loop_type_list[0][index_selected_axis+1:]
            new_loop_notation = curate_loops(values_list[0], keys_list[0], new_loop_type_list)
            new_raw_eq=new_loop_notation+'['+';'.join(equations_under_loops[index])+';];'
            # print(f'new_raw_eq: {new_raw_eq},\n old_eq: {row_equations_under_loops[index]}')
            transformed_IR = ''.join(row_equations_under_loops[:index]) + new_raw_eq + ''.join(row_equations_under_loops[index+1:])
            transformed_IR_list.append(transformed_IR)
            original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_loop_vectorization_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    loop_vectorization_index_list=judge_loop_parallelization_condition(loops, eq_outputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(loop_vectorization_index_list)):
        index, parallel_axis = loop_vectorization_index_list[index]
        this_loop = loops[index]
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        vectorized_axis=[axis for axis in parallel_axis if values_list[0][keys_list[0].index(axis)]<=4]
        if len(vectorized_axis)>0:
            has_transformation = True
            selected_axis = random.choice(vectorized_axis)
            index_selected_axis = keys_list[0].index(selected_axis)
            new_loop_type_list = loop_type_list[0][:index_selected_axis] + ['V'] + loop_type_list[0][index_selected_axis+1:]
            new_loop_notation = curate_loops(values_list[0], keys_list[0], new_loop_type_list)
            new_raw_eq=new_loop_notation+'['+';'.join(equations_under_loops[index])+';];'
            # print(f'new_raw_eq: {new_raw_eq},\n old_eq: {row_equations_under_loops[index]}')
            transformed_IR = ''.join(row_equations_under_loops[:index]) + new_raw_eq + ''.join(row_equations_under_loops[index+1:])
            transformed_IR_list.append(transformed_IR)
            original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_loop_binding_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    loop_binding_index_list=judge_loop_parallelization_condition(loops, eq_outputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(loop_binding_index_list)):
        op_index, parallel_axis = loop_binding_index_list[index]
        this_simplified_eqs= simplified_eqs_under_loops[op_index]
        no_var=True
        for simplified_eq in this_simplified_eqs:
            simplified_eq=simplified_eq.replace('max','').replace('min','').replace('exp','').replace('log','').replace('sqrt','').replace('abs','').replace('if_then_else','').replace('erf','')
            if len(re.findall(rf'\b[a-z]+\b',simplified_eq))>0:
                no_var=False
                break
        if no_var:
            this_loop = loops[op_index]
            values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
            this_eqs= ';'.join(equations_under_loops[op_index])
            can_be_selected_axis=[axis for axis in parallel_axis if values_list[0][keys_list[0].index(axis)]<65535]
            if len(can_be_selected_axis)>0:
                has_transformation = True
                selected_axis = random.choice(can_be_selected_axis)
                loop_idx = keys_list[0].index(selected_axis)
                if values_list[0][loop_idx]*values_list[0][keys_list[0].index('tx')]>1024:
                    selected_bind_location='b'
                else:
                    selected_bind_location=random.choice(['b', 't'])
                if selected_bind_location=='t':
                    if values_list[0][loop_idx]<=64:
                        selected_bind_axis = random.choice(['y', 'z'])
                    else:
                        selected_bind_axis='y'
                else:
                    selected_bind_axis = random.choice(['x', 'y', 'z'])
                # print(f'selected_this_value: {selected_this_value}')
                #this loop: normal loops
                new_subscript=selected_bind_location+selected_bind_axis+keys_list[0][loop_idx]
                new_keys_list = keys_list[0][:loop_idx] + [new_subscript] + keys_list[0][loop_idx+1:]
                new_values_list = values_list[0]
                new_loop_type_list = loop_type_list[0][:loop_idx] + ['B'] + loop_type_list[0][loop_idx+1:]
                original_subscript=keys_list[0][loop_idx]
                new_loop_notation = curate_loops(new_values_list, new_keys_list, new_loop_type_list)
                split_this_eqs_by_subscript = split_eq_by_subscript(this_eqs)
                # print(f'split_this_eqs_by_subscript: {split_this_eqs_by_subscript}')
                replaced_split_eqs=replace_subscript_in_split_eqs(split_this_eqs_by_subscript, original_subscript, new_subscript)
                this_eqs=''.join(replaced_split_eqs)
                transformed_part=new_loop_notation+'['+this_eqs+';];'
                transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                transformed_IR_list.append(transformed_IR)
                original_IR_list.append(IR)
                # print(f'transformed_part: {transformed_part}')
    return original_IR_list, transformed_IR_list, has_transformation

# def apply_loop_binding_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
#     loop_binding_index_list=judge_loop_binding_condition(loops, eq_outputs_under_loops)
#     has_transformation = False
#     transformed_IR_list=[]
#     for index in range(len(loop_binding_index_list)):
#         has_transformation = True
#         op_index, loop_idx, this_value = loop_binding_index_list[index]
#         this_loop = loops[op_index]
#         # print(f'this_loop:{this_loop}\nop_index: {op_index}, loop_idx: {loop_idx}, this_value: {this_value}')
#         values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([loops[op_index]])
#         this_eqs= ','.join(equations_under_loops[op_index])
#         intermediate_keys=generate_idx_names(1, len(keys_list[0])-2)
#         selected_bind_axis = random.choice(['x', 'y', 'z'])
#         selected_this_value=random.choice(this_value)
#         # print(f'selected_this_value: {selected_this_value}')
#         #this loop: normal loops
#         new_keys_list = keys_list[0][:loop_idx] + ['b'+selected_bind_axis+intermediate_keys[0], 't'+selected_bind_axis+keys_list[0][loop_idx]] + keys_list[0][loop_idx+1:]
#         new_values_list = values_list[0][:loop_idx] + [selected_this_value[0], selected_this_value[1]] + values_list[0][loop_idx+1:]
#         new_loop_type_list = loop_type_list[0][:loop_idx] + ['B', 'B'] + loop_type_list[0][loop_idx+1:]
#         original_subscript=keys_list[0][loop_idx]
#         new_subscript='b'+selected_bind_axis+intermediate_keys[0]+'*'+str(selected_this_value[0])+'+t'+selected_bind_axis+keys_list[0][loop_idx]
#         new_loop_notation = curate_loops(new_values_list, new_keys_list, new_loop_type_list)
#         # print(f'this loop: {loops[op_index]},\n new_loop_notation: {new_loop_notation}')
#         # print(f'original_subscript: {original_subscript},\n new_subscript: {new_subscript}')
#         # print(f'this_eqs: {this_eqs}')
#         this_eqs=this_eqs.replace('{'+original_subscript, '{'+new_subscript).replace(','+original_subscript, ','+new_subscript).replace('+'+original_subscript, '+'+new_subscript).replace('*'+original_subscript, '*'+new_subscript)
#         # print(f'modified_this_eqs: {this_eqs}')
#         transformed_part=new_loop_notation+'['+this_eqs+';];'
#         transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
#         transformed_IR_list.append(transformed_IR)
#         # print(f'transformed_part: {transformed_part}')
#     return [IR], transformed_IR_list, has_transformation

def apply_reduction_factorization_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    reduction_factorization_index_list=judge_reduction_factorization_condition(loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops, simplified_eqs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(reduction_factorization_index_list)):
        op_index, eq_idx, split_reduce_aix=reduction_factorization_index_list[index]
        # print(f'op_index: {op_index}, eq_idx: {eq_idx}, split_reduce_aix:{split_reduce_aix}, simplified_eqs_under_loops:{simplified_eqs_under_loops[op_index]}')
        # print(f'eq: {equations_under_loops[op_index][eq_idx]}')
        this_loop = loops[op_index]
        output_superscript = re.findall(r'\^\{.*?\}', eq_outputs_under_loops[op_index][0])[0]
        output_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[op_index][0])
        output_subscript = output_subscript_list[0] if len(output_subscript_list)>0 else ''
        inputs = eq_inputs_under_loops[op_index]
        this_outputs = eq_outputs_under_loops[op_index]
        this_right_simplified_eq='='.join(simplified_eqs_under_loops[op_index][eq_idx].split('=')[1:])
        updated_this_inputs = [item for item in inputs if item not in this_outputs]
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        #for reduce axis
        simplified_eq=simplified_eqs_under_loops[op_index][eq_idx]
        simplified_output=simplified_eq_outputs_under_loops[op_index][0]
        simplified_input=simplified_eq_inputs_under_loops[op_index]
        lower_case= re.findall(r'[a-z]+', simplified_eq.replace('if_then_else', '').replace('erf', '').replace('log', '').replace('sqrt', '').replace('abs', '').replace('exp', '').replace('min', '').replace('max', ''))
        if (len(lower_case)==0) and (len(split_reduce_aix)>0) and (simplified_output+'+' in simplified_eq or simplified_output+'*' in simplified_eq or 'min('+simplified_output in simplified_eq or 'max('+simplified_output in simplified_eq):
            r_axis=random.choice(split_reduce_aix)            
            # print(f'simplified_eq: {simplified_eq}')
            num_cancat_inputs, inputs_name_list, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, loop_notation1, loop_notation2, first_two_loop_notation1,first_two_loop_notation2,new_values_list1=find_different_vars_scripts_and_curate_loops_for_tensor_split_decouple(r_axis, values_list[0], keys_list[0], loop_type_list[0], updated_this_inputs, this_right_simplified_eq)
            if num_cancat_inputs>0:
                intermediate_names, name_start_idx = generate_names(2, name_start_idx)
                concat_values_list=values_list[0][:r_axis]+values_list[0][r_axis+1:]
                concat_keys_list=keys_list[0][:r_axis]+keys_list[0][r_axis+1:]
                new_inputs_subscripts_list=inputs_subscripts_list+[output_subscript]
                new_inputs_superscripts_list=inputs_superscripts_list+[output_superscript]
                output_name=intermediate_names
                if output_subscript=='' and len(concat_values_list)>0:
                    output_subscript='_{'+','.join(concat_keys_list)+'}'
                full_output_subscript=re.findall(r'[a-zA-Z]+|[^a-zA-Z]', output_subscript)
                output_resubscript = [output_subscript, rewrite_subscript_for_cancat(full_output_subscript, [keys_list[0][r_axis]], [r_axis], new_values_list1)]
                # print(f'output_resubscript:{output_resubscript}')
                new_resubscript_list = resubscript_list + [output_resubscript[1]]
                concat_loop_type_list=loop_type_list[0][:r_axis]+loop_type_list[0][r_axis+1:]
                if 'tx' not in concat_keys_list:
                    concat_values_list=['1']+concat_values_list
                    concat_keys_list=['tx']+concat_keys_list
                    concat_loop_type_list=['B']+concat_loop_type_list
                concat_loop_notation=curate_loops(concat_values_list, concat_keys_list, concat_loop_type_list)
                input_same_as_output=[item for item in eq_inputs_under_loops[op_index] if simplified_output+'^' in item]
                if simplified_output not in simplified_input:
                    if simplified_output+'+' in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'='+output_name[0]+output_superscript+output_resubscript[0]+'+'+output_name[1]+output_superscript+output_resubscript[1]+';];'
                    elif simplified_output+'*' in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'='+output_name[0]+output_superscript+output_resubscript[0]+'*'+output_name[1]+output_superscript+output_resubscript[1]+';];'
                    elif 'min('+simplified_output in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'=min('+output_name[0]+output_superscript+output_resubscript[0]+','+output_name[1]+output_superscript+output_resubscript[1]+');];'
                    elif 'max('+simplified_output in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'=max('+output_name[0]+output_superscript+output_resubscript[0]+','+output_name[1]+output_superscript+output_resubscript[1]+');];'
                else:
                    if simplified_output+'+' in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'='+input_same_as_output[0]+'+('+output_name[0]+output_superscript+output_resubscript[0]+'+'+output_name[1]+output_superscript+output_resubscript[1]+');];'
                    elif simplified_output+'*' in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'='+input_same_as_output[0]+'*('+output_name[0]+output_superscript+output_resubscript[0]+'*'+output_name[1]+output_superscript+output_resubscript[1]+');];'
                    elif 'min('+simplified_output in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'=min('+input_same_as_output[0]+',min('+output_name[0]+output_superscript+output_resubscript[0]+','+output_name[1]+output_superscript+output_resubscript[1]+'));];'
                    elif 'max('+simplified_output in simplified_eq:
                        concat_IR=concat_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'=max('+input_same_as_output[0]+',max('+output_name[0]+output_superscript+output_resubscript[0]+','+output_name[1]+output_superscript+output_resubscript[1]+'));];'
                #reference
                new_input_name0, new_input_name1=[], []
                for input_name in inputs_name_list:
                    new_input_name0.append(re.sub(r'\^\{.*?\}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}|\^\{.*?\}','',input_name).replace('}}','}'))
                    new_input_name1.append(re.sub(r'\^\{.*?\}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}|\^\{.*?\}','',input_name).replace('}}','}'))
                if eq_outputs_under_loops[op_index][0] in inputs:
                    new_input_name0.append(output_name[0])
                    new_input_name1.append(output_name[1])
               
                # print(f'output_name: {output_name}, output_resubscript: {output_resubscript}')
                # #decouple the equation
                # print(f'new_inputs_index_list: {new_inputs_index_list}, new_input_name0: {new_input_name0}, new_input_name1: {new_input_name1}')
                expr1=transfrom_from_original_simpified_expr_to_modified_expr('='.join(simplified_eq.split('=')[1:]),simplified_eq_inputs_under_loops[op_index],simplified_eq_outputs_under_loops[op_index], eq_inputs_under_loops[op_index], output_name[0], output_resubscript[0], output_superscript, inputs_name_list, new_input_name0, new_inputs_superscripts_list, new_inputs_subscripts_list)
                expr2=transfrom_from_original_simpified_expr_to_modified_expr('='.join(simplified_eq.split('=')[1:]),simplified_eq_inputs_under_loops[op_index],simplified_eq_outputs_under_loops[op_index], eq_inputs_under_loops[op_index], output_name[1], output_resubscript[0], output_superscript,inputs_name_list, new_input_name1, new_inputs_superscripts_list, new_resubscript_list)
                decouple_IR1=loop_notation1+'['+output_name[0]+output_superscript+output_subscript+'='+expr1+';];'
                decouple_IR2=loop_notation2+'['+output_name[1]+output_superscript+output_subscript+'='+expr2+';];'
                transformed_part = decouple_IR1 + decouple_IR2 + concat_IR 
                transform_IR= ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                # print(f'transform_IR: {transformed_part}')
                if math.prod(concat_values_list)<=2147483648:
                    has_transformation = True
                    transformed_IR_list.append(transform_IR)
                    original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_cache_read_write_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    cache_read_write_index_list=judge_cache_read_write_condition(simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops,simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(cache_read_write_index_list)):
        op_index, can_be_selected_inputs, can_be_selected_outputs = cache_read_write_index_list[index]
        # print(f'op_index: {op_index}, can_be_selected_inputs: {can_be_selected_inputs}, can_be_selected_outputs: {can_be_selected_outputs}')
        this_loop = loops[op_index]
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        # cache_read
        if len(can_be_selected_inputs)>0:
            this_raw_eq= row_equations_under_loops[op_index]
            selected_input1 =random.choice(can_be_selected_inputs)
            # print(f'selected_input: {selected_input1}')
            selected_input_memory_location= random.choice(['s', 'l'])
            input_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
            input_superscript = re.findall(r'\^\{.*?\}', selected_input1)[0]
            new_input_superscript = input_superscript.replace(',g}', ','+selected_input_memory_location+'}')
            input_full_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(selected_input1)
            input_full_subscript = input_full_subscript_list[0]
            minus_num_list=re.findall(rf'-[0-9]+|[0-9]+-',input_full_subscript)
            if len(minus_num_list)==0:
                new_input1=input_intermediate_names[0]+new_input_superscript+input_full_subscript
                input_subscript_list=re.findall(r'[a-zA-Z]+', input_full_subscript)
                input_loop_index_list=[idx for idx in range(len(keys_list[0])) if keys_list[0][idx] in input_subscript_list]
                input_loop_index_list.sort()
                new_input_keys_list = [keys_list[0][idx] for idx in input_loop_index_list]
                new_input_values_list = [values_list[0][idx] for idx in input_loop_index_list]
                new_input_loop_type_list = [loop_type_list[0][idx] for idx in input_loop_index_list]
                new_input2=new_input1
                selected_input2=selected_input1
                has_tx=True
                if 'tx' not in new_input_keys_list:
                    can_be_selected_axis=[idx for idx in range(len(new_input_values_list)) if new_input_values_list[idx]<1024]
                    if len(can_be_selected_axis)>0:
                        has_tx=True
                        axis=can_be_selected_axis[0]
                        original_input_first_key=new_input_keys_list[axis]
                        modified_input_first_key='tx'
                        split_new_input2_by_subscript = split_eq_by_subscript(new_input2)
                        split_selected_input2_by_subscript = split_eq_by_subscript(selected_input2)
                        replaced_split_new_input2_eqs=replace_subscript_in_split_eqs(split_new_input2_by_subscript, original_input_first_key, modified_input_first_key)
                        replaced_split_selected_input2_eqs=replace_subscript_in_split_eqs(split_selected_input2_by_subscript, original_input_first_key, modified_input_first_key)
                        new_input2=''.join(replaced_split_new_input2_eqs)
                        selected_input2=''.join(replaced_split_selected_input2_eqs)
                        # print(f'new_input2:{new_input2}, selected_input2:{selected_input2}')
                        new_input_keys_list = new_input_keys_list[:axis]+['tx'] + new_input_keys_list[axis+1:]
                        new_input_loop_type_list = new_input_loop_type_list[:axis]+['B'] + new_input_loop_type_list[axis+1:]
                    else:
                        has_tx=False
                if has_tx:
                    has_transformation = True
                    input_loop_notation = curate_loops(new_input_values_list, new_input_keys_list, new_input_loop_type_list)
                    new_input_eq= input_loop_notation + '[' + new_input2 + '=' + selected_input2 + ';];'
                    new_this_eq=this_raw_eq.replace(selected_input1, new_input1)
                    transformed_part = new_input_eq + new_this_eq
                    transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                    transformed_IR_list.append(transformed_IR)
                    original_IR_list.append(IR)
                    # print(f'original IR:{row_equations_under_loops[op_index]}\ntransformed_part: {transformed_part}')
        # cache_write
        if len(can_be_selected_outputs)>0:
            this_raw_eq= row_equations_under_loops[op_index]
            selected_output1 = random.choice(can_be_selected_outputs)
            # print(f'selected_output: {selected_output1}')
            selected_output_memory_location = random.choice(['s', 'l'])
            output_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
            output_superscript = re.findall(r'\^\{.*?\}', selected_output1)[0]
            new_output_superscript = output_superscript.replace(',g}', ','+selected_output_memory_location+'}')
            output_full_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(selected_output1)
            output_full_subscript = output_full_subscript_list[0]
            minus_num_list=re.findall(rf'-[0-9]+|[0-9]+-',output_full_subscript)
            if len(minus_num_list)==0:
                new_output1= output_intermediate_names[0] + new_output_superscript + output_full_subscript
                output_subscript_list = re.findall(r'[a-zA-Z]+', output_full_subscript)
                output_loop_index_list = [idx for idx in range(len(keys_list[0])) if keys_list[0][idx] in output_subscript_list]
                new_output_keys_list = [keys_list[0][idx] for idx in output_loop_index_list]
                new_output_values_list = [values_list[0][idx] for idx in output_loop_index_list]
                new_output_loop_type_list = [loop_type_list[0][idx] for idx in output_loop_index_list]
                new_output2=new_output1
                selected_output2=selected_output1
                has_tx=True
                if 'tx' not in new_output_keys_list:
                    can_be_selected_axis=[idx for idx in range(len(new_output_keys_list)) if new_output_keys_list[idx]<1024]
                    if len(can_be_selected_axis)>0:
                        has_tx=True
                        axis=can_be_selected_axis[0]
                        original_output_first_key=new_output_keys_list[axis]
                        modified_output_first_key='tx'
                        split_new_output2_by_subscript = split_eq_by_subscript(new_output2)
                        split_selected_output2_by_subscript = split_eq_by_subscript(selected_output2)
                        replaced_split_new_output2_eqs=replace_subscript_in_split_eqs(split_new_output2_by_subscript, original_output_first_key, modified_output_first_key)
                        replaced_split_selected_output2_eqs=replace_subscript_in_split_eqs(split_selected_output2_by_subscript, original_output_first_key, modified_output_first_key)
                        new_output2=''.join(replaced_split_new_output2_eqs)
                        selected_output2=''.join(replaced_split_selected_output2_eqs)
                        # print(f'new_output2:{new_output2}, selected_output2:{selected_output2}')
                        new_output_keys_list = new_output_keys_list[:axis]+['tx']+new_output_keys_list[axis+1:]
                        new_output_loop_type_list =new_output_loop_type_list[:axis]+['B']+new_output_loop_type_list[axis+1:]
                    else:
                        has_tx=False
                if has_tx:
                    has_transformation = True
                    output_loop_notation = curate_loops(new_output_values_list, new_output_keys_list, new_output_loop_type_list)
                    new_this_eq = this_raw_eq.replace(selected_output1, new_output1)
                    new_output_eq = output_loop_notation + '[' + selected_output2 + '=' + new_output2 + ';];'
                    transformed_part = new_this_eq + new_output_eq
                    transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                    transformed_IR_list.append(transformed_IR)
                    original_IR_list.append(IR)
                    # print(f'original IR:{row_equations_under_loops[op_index]}\ntransformed_part: {transformed_part}')
    return original_IR_list, transformed_IR_list, has_transformation

def apply_layout_transformation_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    #transformation types: 
    # 1.transpose(multiple dims(>1)->multiple dims, same number of dim) 
    # 2. reshape(multiple dims/single dim(>=1, factorization)->more dims, multiple dims (>1)->less dims)
    # 3. flatten (multiple dims(>1)->single dim)
    # 4. squeeze (multiple dims including 1(>1)-> less dims)
    layout_transformation_index_list=judge_layout_transformation_condition(loops,simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    original_IR_list=[]
    for index in range(len(layout_transformation_index_list)):
        op_index, candidate_inputs_transpose_flatten_reshape2, candidate_inputs_reshape1, candidate_inputs_squeeze=layout_transformation_index_list[index]
        # print(f'op_index: {op_index}\ncandidate_inputs_transpose_flatten_reshape2: {candidate_inputs_transpose_flatten_reshape2}\ncandidate_inputs_reshape1: {candidate_inputs_reshape1}\ncandidate_inputs_squeeze: {candidate_inputs_squeeze}')
        # print(f'op_index: {op_index}\ncandidate_inputs_squeeze: {candidate_inputs_squeeze}')
        this_loop = loops[op_index]
        this_raw_eq = row_equations_under_loops[op_index]
        # print(f'this_raw_eq: {this_raw_eq}')
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        intermediate_names, name_start_idx = generate_names(1, name_start_idx)
        key_value_mapping={keys_list[0][idx]: values_list[0][idx] for idx in range(len(keys_list[0]))}
        # print(f'key_value_mapping:{key_value_mapping}')
        if len(candidate_inputs_transpose_flatten_reshape2)>0:
            for idx in range(len(candidate_inputs_transpose_flatten_reshape2)):
                this_input, subscript_details=candidate_inputs_transpose_flatten_reshape2[idx]
                key_value_mapping={keys_list[0][idx]: values_list[0][idx]-1 for idx in range(len(keys_list[0]))}
                subscript_details_value =[eval(expr, {}, key_value_mapping)+1 for expr in subscript_details]
                # print(f'subscript_details_value:{subscript_details_value}')
                original_subscript='_{'+','.join(subscript_details)+'}'
                this_input_superscript= re.findall(r'\^{[a-zA-Z0-9,]*}', this_input)[0]
                minus_num_list=re.findall(rf'-[0-9]+|[0-9]+-',original_subscript)
                this_input_keys= re.findall(r'[a-zA-Z]+', original_subscript)
                original_idx_this_input_keys_in_key_list=[idx for idx in range(len(keys_list[0])) if keys_list[0][idx] in this_input_keys]
                if len(minus_num_list)==0:
                    #transpose
                    if len(subscript_details)>2:
                        shuffled_subscript_details = random_shuffule(subscript_details)
                    else:
                        shuffled_subscript_details=[subscript_details[1],subscript_details[0]]
                    shuffled_subscript_in_eq = '_{' + ','.join(shuffled_subscript_details) + '}'
                    new_transpose_input_in_eq = intermediate_names[0]+ this_input_superscript + shuffled_subscript_in_eq
                    new_transpose_loop_idx=original_idx_this_input_keys_in_key_list.copy()
                    new_transpose_loop_idx.sort()
                    new_transpose_key_list = [keys_list[0][idx] for idx in new_transpose_loop_idx]
                    new_transpose_value_list = [values_list[0][idx] for idx in new_transpose_loop_idx]
                    new_transpose_loop_type_list = [loop_type_list[0][idx] for idx in new_transpose_loop_idx]
                    new_transpose_input_in_transpose=new_transpose_input_in_eq
                    this_input_in_transpose = this_input
                    has_tx=True
                    if 'tx' not in this_input_keys:
                        can_be_selected_axis=[idx for idx in range(len(new_transpose_value_list)) if new_transpose_value_list[idx]<1024]
                        if len(can_be_selected_axis)>0:
                            has_tx=True
                            axis=can_be_selected_axis[0]
                            original_output_first_key=new_transpose_key_list[axis]
                            modified_output_first_key='tx'
                            split_new_transpose_input_in_transpose_by_subscript = split_eq_by_subscript(new_transpose_input_in_transpose)
                            split_this_input_in_transpose_by_subscript = split_eq_by_subscript(this_input_in_transpose)
                            replaced_split_new_transpose_input_in_transpose_eqs=replace_subscript_in_split_eqs(split_new_transpose_input_in_transpose_by_subscript, original_output_first_key, modified_output_first_key)
                            replaced_this_input_in_transpose_eqs=replace_subscript_in_split_eqs(split_this_input_in_transpose_by_subscript, original_output_first_key, modified_output_first_key)
                            new_transpose_input_in_transpose=''.join(replaced_split_new_transpose_input_in_transpose_eqs)
                            this_input_in_transpose=''.join(replaced_this_input_in_transpose_eqs)
                            new_transpose_key_list = new_transpose_key_list[:axis] + ['tx'] + new_transpose_key_list[axis+1:]
                            new_transpose_loop_type_list = new_transpose_loop_type_list[:axis] + ['B'] + new_transpose_loop_type_list[axis+1:]
                        else:
                            has_tx=False
                    if has_tx:
                        has_transformation = True
                        new_transpose_loop_notation = curate_loops(new_transpose_value_list, new_transpose_key_list, new_transpose_loop_type_list)
                        transpose_eq=new_transpose_loop_notation+'['+new_transpose_input_in_transpose+'='+this_input_in_transpose+';];'
                        new_raw_transpose_eq = this_raw_eq.replace(this_input, new_transpose_input_in_eq)
                        transpose_transformed_part = transpose_eq+new_raw_transpose_eq
                        transpose_transform_IR= ''.join(row_equations_under_loops[:op_index]) + transpose_transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                        transformed_IR_list.append(transpose_transform_IR)
                        original_IR_list.append(IR)
                        # print(f'this_input:{this_input}\nthis_raw_eq:{this_raw_eq}\ntranspose_transformed_part: {transpose_transformed_part}')
                    #flatten
                    flatten_subscript_details=subscript_details
                    this_input_in_flatten=this_input
                    new_flatten_key_list = [keys_list[0][idx] for idx in original_idx_this_input_keys_in_key_list]
                    new_flatten_value_list = [values_list[0][idx] for idx in original_idx_this_input_keys_in_key_list]
                    new_flatten_loop_type_list = [loop_type_list[0][idx] for idx in original_idx_this_input_keys_in_key_list]
                    has_tx=True
                    if 'tx' not in this_input_keys:
                        can_be_selected_axis=[idx for idx in range(len(new_flatten_key_list)) if new_flatten_value_list[idx]<1024]
                        if len(can_be_selected_axis)>0:
                            has_tx=True
                            axis=can_be_selected_axis[0]
                            original_output_first_key=new_flatten_key_list[axis]
                            modified_output_first_key='tx'
                            flatten_subscript_details_var_list=['_{'+','.join(flatten_subscript_details)+'}']
                            split_this_input_in_flatten_by_subscript = split_eq_by_subscript(this_input_in_flatten)
                            replaced_split_flatten_subscript_details_var_list_eqs=replace_subscript_in_split_eqs(flatten_subscript_details_var_list, original_output_first_key, modified_output_first_key)
                            replaced_split_this_input_in_flatten_eqs=replace_subscript_in_split_eqs(split_this_input_in_flatten_by_subscript, original_output_first_key, modified_output_first_key)
                            flatten_subscript_details_var=replaced_split_flatten_subscript_details_var_list_eqs[0]
                            flatten_subscript_details=flatten_subscript_details_var.replace('_{','').replace('}','').split(',')
                            this_input_in_flatten=''.join(replaced_split_this_input_in_flatten_eqs)
                            new_flatten_key_list = new_flatten_key_list[:axis] + ['tx'] + new_flatten_key_list[axis+1:]
                            new_flatten_loop_type_list = new_flatten_loop_type_list[:axis] + ['B'] + new_flatten_loop_type_list[axis+1:]
                        else:
                            has_tx=False
                    if has_tx:
                        has_transformation = True
                        flatten_loop_notation= curate_loops(new_flatten_value_list, new_flatten_key_list, new_flatten_loop_type_list)
                        flatten_subscript_details_in_eq=''
                        flatten_subscript_details_in_flatten=''
                        len_subscript=len(subscript_details)
                        for idx in range(len_subscript-1):
                            if math.prod(subscript_details_value[idx+1:])>1:
                                if len(re.sub('[a-z]+','', subscript_details[idx]))==0:
                                    flatten_subscript_details_in_eq+=subscript_details[idx]+'*'+str(math.prod(subscript_details_value[idx+1:]))+'+'
                                elif not subscript_details[idx].isdigit():
                                    flatten_subscript_details_in_eq+='('+subscript_details[idx]+')*'+str(math.prod(subscript_details_value[idx+1:]))+'+'
                                if len(re.sub('[a-z]+','', flatten_subscript_details[idx]))==0:
                                    flatten_subscript_details_in_flatten+=flatten_subscript_details[idx]+'*' +str(math.prod(subscript_details_value[idx+1:]))+'+'
                                elif not flatten_subscript_details[idx].isdigit():
                                    flatten_subscript_details_in_flatten+='('+flatten_subscript_details[idx]+')*'+str(math.prod(subscript_details_value[idx+1:]))+'+'
                            else:
                                if len(re.sub('[a-z]+','', subscript_details[idx]))==0:
                                    flatten_subscript_details_in_eq+=subscript_details[idx]+'+'
                                elif not subscript_details[idx].isdigit():
                                    flatten_subscript_details_in_eq+='('+subscript_details[idx]+')+'
                                if len(re.sub('[a-z]+','', flatten_subscript_details[idx]))==0:
                                    flatten_subscript_details_in_flatten+=flatten_subscript_details[idx]+'+'
                                elif not flatten_subscript_details[idx].isdigit():
                                    flatten_subscript_details_in_flatten+='('+flatten_subscript_details[idx]+')+'
                        if len(re.sub('[a-z]+','', subscript_details[len_subscript-1]))==0:
                            flatten_subscript_details_in_eq+=subscript_details[len_subscript-1] 
                        elif not subscript_details[len_subscript-1].isdigit():
                            flatten_subscript_details_in_eq+='('+subscript_details[len_subscript-1]+')'
                        else:
                            flatten_subscript_details_in_eq=flatten_subscript_details_in_eq[:-1]
                        if len(re.sub('[a-z]+','', flatten_subscript_details[len_subscript-1]))==0:
                            flatten_subscript_details_in_flatten+=flatten_subscript_details[len_subscript-1]  
                        elif not flatten_subscript_details[len_subscript-1].isdigit():
                            flatten_subscript_details_in_flatten+='('+flatten_subscript_details[len_subscript-1]+')'
                        else:
                            flatten_subscript_details_in_flatten=flatten_subscript_details_in_flatten[:-1]
                        # print(f'flatten_subscript_details_in_eq: {flatten_subscript_details_in_eq}, flatten_subscript_details_in_flatten: {flatten_subscript_details_in_flatten}')
                        flatten_input_in_eq = intermediate_names[0] + this_input_superscript + '_{' + flatten_subscript_details_in_eq+ '}'
                        flatten_input_in_flatten = intermediate_names[0] + this_input_superscript + '_{' + flatten_subscript_details_in_flatten + '}'
                        flatten_eq = flatten_loop_notation + '[' + flatten_input_in_flatten + '=' + this_input_in_flatten + ';];'
                        new_raw_flatten_eq = this_raw_eq.replace(this_input, flatten_input_in_eq)
                        flatten_transformed_part = flatten_eq + new_raw_flatten_eq
                        flatten_transform_IR = ''.join(row_equations_under_loops[:op_index]) + flatten_transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                        transformed_IR_list.append(flatten_transform_IR)
                        original_IR_list.append(IR)
                        # print(f'this_input:{this_input}\nthis_raw_eq:{this_raw_eq}\nflatten_transformed_part: {flatten_transformed_part}')
                    #reshape2
                    reshape2_selected_axis = random.sample(subscript_details, 2)
                    reshape2_selected_axis_index = [subscript_details.index(item) for item in reshape2_selected_axis]
                    reshape2_selected_axis_value=[subscript_details_value[idx] for idx in reshape2_selected_axis_index]
                    left_reshape2_subscript_list=list(set(subscript_details)-set(reshape2_selected_axis))
                    non_digital_subscript_index_in_left_reshape2=[idx for idx in range(len(left_reshape2_subscript_list)) if not left_reshape2_subscript_list[idx].isdigit()]
                    if len(left_reshape2_subscript_list)>2 and len(non_digital_subscript_index_in_left_reshape2)>2:
                        fused_reshape2_axis=''
                        if  len(re.sub('[a-z]+','', reshape2_selected_axis[0]))==0:
                            fused_reshape2_axis+=reshape2_selected_axis[0]+'*'+str(reshape2_selected_axis_value[1])+'+'  
                        elif not reshape2_selected_axis[0].isdigit():
                            fused_reshape2_axis+='('+reshape2_selected_axis[0]+')*'+str(reshape2_selected_axis_value[1])+'+'
                        if len(re.sub('[a-z]+','', reshape2_selected_axis[1]))==0:
                            fused_reshape2_axis+=reshape2_selected_axis[1]
                        elif not reshape2_selected_axis[1].isdigit():
                            fused_reshape2_axis+='('+reshape2_selected_axis[1]+')'
                        if fused_reshape2_axis!='':
                            temp_reshape2_subscript_list = left_reshape2_subscript_list + [fused_reshape2_axis]
                        else:
                            temp_reshape2_subscript_list = left_reshape2_subscript_list
                        if len(temp_reshape2_subscript_list)>2:
                            reshape2_subscript_list_in_eq = random_shuffule(temp_reshape2_subscript_list)
                        else:
                            reshape2_subscript_list_in_eq=[temp_reshape2_subscript_list[1],temp_reshape2_subscript_list[0]]
                        reshape2_subscript_list_in_reshape2=reshape2_subscript_list_in_eq
                        this_input_in_reshape2=this_input
                        new_reshape2_key_list=[keys_list[0][idx] for idx in original_idx_this_input_keys_in_key_list]
                        new_reshape2_value_list=[values_list[0][idx] for idx in original_idx_this_input_keys_in_key_list]
                        new_reshape2_loop_type_list=[loop_type_list[0][idx] for idx in original_idx_this_input_keys_in_key_list]
                        has_tx=True
                        if 'tx' not in this_input_keys:
                            can_be_selected_axis=[idx for idx in range(len(new_reshape2_key_list)) if new_reshape2_value_list[idx]<1024]
                            if len(can_be_selected_axis)>0:
                                has_tx=True
                                axis=can_be_selected_axis[0]
                                original_output_first_key=new_reshape2_key_list[axis]
                                modified_output_first_key='tx'
                                reshape2_subscript_list_in_reshape2_var_list=['_{'+','.join(reshape2_subscript_list_in_reshape2)+'}']
                                split_this_input_in_reshape2_by_subscript = split_eq_by_subscript(this_input_in_reshape2)
                                replaced_split_reshape2_subscript_list_in_reshape2_var_list_eqs=replace_subscript_in_split_eqs(reshape2_subscript_list_in_reshape2_var_list, original_output_first_key, modified_output_first_key)
                                replaced_split_this_input_in_reshape2_eqs=replace_subscript_in_split_eqs(split_this_input_in_reshape2_by_subscript, original_output_first_key, modified_output_first_key)
                                replaced_split_reshape2_subscript_list_in_reshape2_var=replaced_split_reshape2_subscript_list_in_reshape2_var_list_eqs[0]
                                reshape2_subscript_list_in_reshape2=replaced_split_reshape2_subscript_list_in_reshape2_var.replace('_{','').replace('}','').split(',')
                                this_input_in_reshape2=''.join(replaced_split_this_input_in_reshape2_eqs)
                                new_reshape2_key_list = new_reshape2_key_list[:axis] + ['tx'] + new_reshape2_key_list[axis+1:]
                                new_reshape2_loop_type_list = new_reshape2_loop_type_list[:axis] + ['B'] + new_reshape2_loop_type_list[axis+1:]
                            else:
                                has_tx=False
                        if has_tx:
                            has_transformation=True
                            new_reshape2_loop_notation = curate_loops(new_reshape2_value_list, new_reshape2_key_list, new_reshape2_loop_type_list)
                            reshape2_input_in_eq = intermediate_names[0] + this_input_superscript + '_{' + ','.join(reshape2_subscript_list_in_eq) + '}'
                            reshape2_input_in_reshape2 = intermediate_names[0] + this_input_superscript + '_{' + ','.join(reshape2_subscript_list_in_reshape2) + '}'
                            reshape2_eq = new_reshape2_loop_notation + '[' + reshape2_input_in_reshape2 + '=' + this_input_in_reshape2 + ';];'
                            new_raw_reshape2_eq = this_raw_eq.replace(this_input, reshape2_input_in_eq)
                            reshape2_transformed_part = reshape2_eq + new_raw_reshape2_eq
                            reshape2_transform_IR = ''.join(row_equations_under_loops[:op_index]) + reshape2_transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                            transformed_IR_list.append(reshape2_transform_IR)
                            original_IR_list.append(IR)
                            # print(f'this_input:{this_input}\nthis_raw_eq:{this_raw_eq}\nreshape2_transformed_part: {reshape2_transformed_part}')
        if len(candidate_inputs_reshape1)>0:
            #reshape(multiple dims/single dim(>=1, factorization)->more dims
            for idx in range(len(candidate_inputs_reshape1)):
                this_input, subscript_details, subscript_details_value, subscript_idx, this_value=candidate_inputs_reshape1[idx]
                # print(f'this_input: {this_input}\nsubscript_details: {subscript_details}\nsubscript_details_value:{subscript_details_value}\nsubscript_idx: {subscript_idx}\nthis_value: {this_value}')
                original_subscript='_{'+','.join(subscript_details)+'}'
                minus_num_list=re.findall(rf'-[0-9]+|[0-9]+-',original_subscript)
                if len(minus_num_list)==0:
                    this_input_superscript= re.findall(r'\^{[a-zA-Z0-9,]*}', this_input)[0]
                    this_input_keys= re.findall(r'[a-zA-Z]+', original_subscript)
                    # print(f'this_input_keys:{this_input_keys}')
                    subscript_idx_in_this_input_keys=this_input_keys.index(subscript_details[subscript_idx])
                    # print(f'subscript_idx_in_this_input_keys:{subscript_idx_in_this_input_keys}')
                    original_idx_this_input_keys_in_key_list=[idx for key in this_input_keys for idx in range(len(keys_list[0])) if keys_list[0][idx] in key]
                    # print(f'original_idx_this_input_keys_in_key_list:{original_idx_this_input_keys_in_key_list}')
                    reshape1_factorization_list=random.choice(this_value)
                    reshape1_subscript_name=generate_idx_names(1, len(keys_list[0])+1)
                    subscript_details_in_eq=subscript_details[:subscript_idx]+[reshape1_subscript_name[0]]+ subscript_details[subscript_idx:]
                    reshape1_output_subscript_list_in_reshape1=subscript_details_in_eq
                    reshape1_output_subscript_list_in_eq=subscript_details[:subscript_idx]+[subscript_details[subscript_idx]+'//'+str(reshape1_factorization_list[1]),subscript_details[subscript_idx]+'%'+str(reshape1_factorization_list[1])]+ subscript_details[subscript_idx+1:]
                    reshape1_input_subscript_list_in_reshape1=subscript_details_in_eq[:subscript_idx]+[subscript_details_in_eq[subscript_idx]+'*'+str(reshape1_factorization_list[1])+'+'+subscript_details_in_eq[subscript_idx+1]]+ subscript_details_in_eq[subscript_idx+2:]
                    new_reshape1_keys_list_in_reshape1=[keys_list[0][idx] for idx in original_idx_this_input_keys_in_key_list[:subscript_idx_in_this_input_keys]] + [reshape1_subscript_name[0]]+[keys_list[0][idx] for idx in original_idx_this_input_keys_in_key_list[subscript_idx_in_this_input_keys:]]
                    new_reshape1_loop_types_list_in_reshape1=[loop_type_list[0][idx] for idx in original_idx_this_input_keys_in_key_list[:subscript_idx_in_this_input_keys]] + ['L']+[loop_type_list[0][idx] for idx in original_idx_this_input_keys_in_key_list[subscript_idx_in_this_input_keys:]]
                    new_reshape1_value_list_in_reshape1=[values_list[0][idx] for idx in original_idx_this_input_keys_in_key_list[:subscript_idx_in_this_input_keys]] + [reshape1_factorization_list[0],reshape1_factorization_list[1]]+[values_list[0][idx] for idx in original_idx_this_input_keys_in_key_list[subscript_idx_in_this_input_keys+1:]]
                    has_tx=True
                    if 'tx' not in this_input_keys:
                        can_be_selected_axis=[idx for idx in range(len(new_reshape1_keys_list_in_reshape1)) if new_reshape1_value_list_in_reshape1[idx]<1024]
                        if len(can_be_selected_axis)>0:
                            has_tx=True
                            axis=can_be_selected_axis[0]
                            original_output_first_key=new_reshape1_keys_list_in_reshape1[axis]
                            modified_output_first_key='tx'
                            reshape1_output_subscript_list_in_reshape1_var_list=['_{'+','.join(reshape1_output_subscript_list_in_reshape1)+'}']
                            reshape1_input_subscript_list_in_reshape1_var_list=['_{'+','.join(reshape1_input_subscript_list_in_reshape1)+'}']
                            replaced_split_reshape1_output_subscript_list_in_reshape1_var_list_eqs=replace_subscript_in_split_eqs(reshape1_output_subscript_list_in_reshape1_var_list, original_output_first_key, modified_output_first_key)
                            replaced_split_reshape1_input_subscript_list_in_reshape1_var_list_eqs=replace_subscript_in_split_eqs(reshape1_input_subscript_list_in_reshape1_var_list, original_output_first_key, modified_output_first_key)
                            reshape1_output_subscript_list_in_reshape1_var=replaced_split_reshape1_output_subscript_list_in_reshape1_var_list_eqs[0]
                            reshape1_output_subscript_list_in_reshape1=reshape1_output_subscript_list_in_reshape1_var.replace('_{','').replace('}','').split(',')
                            reshape1_input_subscript_list_in_reshape1_var=replaced_split_reshape1_input_subscript_list_in_reshape1_var_list_eqs[0]
                            reshape1_input_subscript_list_in_reshape1=reshape1_input_subscript_list_in_reshape1_var.replace('_{','').replace('}','').split(',')
                            new_reshape1_keys_list_in_reshape1 = new_reshape1_keys_list_in_reshape1[:axis] + ['tx'] + new_reshape1_keys_list_in_reshape1[axis+1:]
                            new_reshape1_loop_types_list_in_reshape1 = new_reshape1_loop_types_list_in_reshape1[:axis] + ['B'] + new_reshape1_loop_types_list_in_reshape1[axis+1:]
                        else:
                            has_tx=False
                    if has_tx:
                        has_transformation = True
                        reshape1_input_in_eq = intermediate_names[0] + this_input_superscript + '_{' + ','.join(reshape1_output_subscript_list_in_eq) + '}'
                        reshape1_output_in_reshape1 = intermediate_names[0] + this_input_superscript + '_{' + ','.join(reshape1_output_subscript_list_in_reshape1) + '}'
                        reshape1_input_in_reshape1 = this_input.replace(original_subscript, '_{' + ','.join(reshape1_input_subscript_list_in_reshape1) + '}')
                        reshape1_loop_notation_in_reshape1 = curate_loops(new_reshape1_value_list_in_reshape1, new_reshape1_keys_list_in_reshape1, new_reshape1_loop_types_list_in_reshape1)
                        # print(f'reshape1_loop_notation_in_reshape1:{reshape1_loop_notation_in_reshape1}')
                        reshape1_eq_in_reshape1 = reshape1_loop_notation_in_reshape1 + '[' + reshape1_output_in_reshape1 + '=' + reshape1_input_in_reshape1 +';];'
                        new_raw_reshape1_eq=this_loop+'['+';'.join(equations_under_loops[op_index]).replace(this_input, reshape1_input_in_eq)+';];'
                        reshape1_transformed_part = reshape1_eq_in_reshape1 + new_raw_reshape1_eq
                        reshape1_transform_IR = ''.join(row_equations_under_loops[:op_index]) + reshape1_transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                        transformed_IR_list.append(reshape1_transform_IR)
                        original_IR_list.append(IR)
                        # print(f'this_input:{this_input}\nthis_raw_eq:{this_raw_eq}\nreshape1_transformed_part: {reshape1_transformed_part}')
    return  original_IR_list, transformed_IR_list, has_transformation

def apply_memory_coalescing_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    #memory coalescing
    memory_coalescing_index_list = judge_memory_coalescing_condition(loops, simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    for index in range(len(memory_coalescing_index_list)):
        op_index, output_or_input, subscript_details, subscript_details_value, coalescing_subscript_details= memory_coalescing_index_list[index]
        # print(f'op_index: {op_index}\noutput_or_input: {output_or_input}\nsubscript_details: {subscript_details}\nsubscript_details_value:{subscript_details_value}\ncoalescing_subscript_details: {coalescing_subscript_details}')
        this_loop = loops[op_index]
        this_eqs = ';'.join(equations_under_loops[op_index])
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        alpha_subscript_index=[idx for idx in range(len(subscript_details)) if subscript_details[idx].isalpha()]
        for coalescing_subscripts in coalescing_subscript_details:
            coalescing_subscripts_idx_in_subscript_details=subscript_details.index(coalescing_subscripts)
            if 'bx' in coalescing_subscripts:
                if len(alpha_subscript_index)>0:
                    has_transformation = True
                    bx_index_in_keys_list=keys_list[0].index('bx')
                    tx_index_in_keys_list=keys_list[0].index('tx')
                    # print(f'subscript_details[alpha_subscript_index[0]:{subscript_details[alpha_subscript_index[0]]}')
                    first_alpha_index_in_keys_list=keys_list[0].index(subscript_details[alpha_subscript_index[0]])
                    new_key_list=keys_list[0].copy()
                    new_key_list[tx_index_in_keys_list]='bx'
                    new_key_list[first_alpha_index_in_keys_list]='tx'
                    new_key_list=new_key_list[:bx_index_in_keys_list]+new_key_list[bx_index_in_keys_list+1:]
                    new_value_list=values_list[0][:bx_index_in_keys_list]+values_list[0][bx_index_in_keys_list+1:]
                    new_loop_type_list=loop_type_list[0].copy()
                    new_loop_type_list[first_alpha_index_in_keys_list]='B'
                    new_loop_type_list=new_loop_type_list[:bx_index_in_keys_list]+new_loop_type_list[bx_index_in_keys_list+1:]
                    new_this_eq=this_eqs.replace('{'+coalescing_subscripts, '{bx').replace(','+coalescing_subscripts, ',bx').replace('+'+coalescing_subscripts, '+bx').replace('*'+coalescing_subscripts, '*bx')
                    new_this_eq= new_this_eq.replace('{'+subscript_details[alpha_subscript_index[0]], '{tx').replace(','+subscript_details[alpha_subscript_index[0]], ',tx').replace('+'+subscript_details[alpha_subscript_index[0]], '+tx').replace('*'+subscript_details[alpha_subscript_index[0]], '*tx')
            else:
                has_transformation = True
                new_subscript_name=random.choice(['bx','by','bz', 'tx', 'ty', 'tz'])+generate_idx_names(1, 0)[0]
                new_this_eq=this_eqs.replace('{'+coalescing_subscripts,'{'+new_subscript_name).replace(','+coalescing_subscripts,','+new_subscript_name).replace('+'+coalescing_subscripts,'+'+new_subscript_name).replace('*'+coalescing_subscripts,'*'+new_subscript_name)
                new_index_list=[idx for idx in range(len(keys_list[0])) if ('{'+keys_list[0][idx] in new_this_eq) or (','+keys_list[0][idx] in new_this_eq) or ('+'+keys_list[0][idx] in new_this_eq) or ('*'+keys_list[0][idx] in new_this_eq)]
                new_index_list.sort()
                new_key_list = [keys_list[0][idx] for idx in new_index_list]+[new_subscript_name]
                new_loop_type_list = [loop_type_list[0][idx] for idx in new_index_list]+['B']
                new_value_list = [values_list[0][idx] for  idx in new_index_list] + [subscript_details_value[coalescing_subscripts_idx_in_subscript_details]]
            new_loop_notation = curate_loops(new_value_list, new_key_list, new_loop_type_list)
            transformed_part = new_loop_notation + '[' +new_this_eq + ';];'
            transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
            transformed_IR_list.append(transformed_IR)
            # print(f'coalescing_subscripts:{coalescing_subscripts}, new_subscript_details: {new_subscript_details}\nthis_eqs:{this_eqs}\nnew_this_eq: {new_this_eq}')
            # print(f'raw_eq: {row_equations_under_loops[op_index]}\ntransformed_part: {transformed_part}')
    return [IR], transformed_IR_list, has_transformation

def apply_vectorized_memory_access_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    vectorized_memory_access_index_list=judge_vectorized_memory_access_condition(simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list=[]
    for index in range(len(vectorized_memory_access_index_list)):
        op_index, candidate_inputs, candidate_outputs = vectorized_memory_access_index_list[index]
        # print(f'op_index: {op_index}, candidate_inputs: {candidate_inputs}, candidate_outputs: {candidate_outputs}')
        this_loop = loops[op_index]
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        # cache_read
        if len(candidate_inputs)>0:
            this_raw_eq= row_equations_under_loops[op_index]
            selected_input1 =random.choice(candidate_inputs)
            # print(f'selected_input: {selected_input1}')
            selected_input_memory_location= random.choice(['s', 'l'])
            input_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
            input_superscript = re.findall(r'\^\{.*?\}', selected_input1)[0]
            new_input_superscript = input_superscript.replace(',g}', ','+selected_input_memory_location+'}')
            input_full_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(selected_input1)
            input_full_subscript = input_full_subscript_list[0]
            new_input1=input_intermediate_names[0]+new_input_superscript+input_full_subscript
            input_subscript_list=re.findall(r'[a-zA-Z]+', input_full_subscript)
            input_loop_index_list=[idx for idx in range(len(keys_list[0])) if keys_list[0][idx] in input_subscript_list]
            input_loop_index_list.sort()
            new_input_keys_list = [keys_list[0][idx] for idx in input_loop_index_list]
            new_input_values_list = [values_list[0][idx] for idx in input_loop_index_list]
            new_input_loop_type_list = [loop_type_list[0][idx] for idx in input_loop_index_list]
            new_input2=new_input1
            selected_input2=selected_input1
            if 'bx' not in new_input_keys_list or 'tx' not in new_input_keys_list:
                original_input_first_key=new_input_keys_list[0]
                modified_input_first_key='bx*'+str(new_input_values_list[0])+'+tx'
                new_input2 = new_input2.replace('_{'+original_input_first_key,'_{'+modified_input_first_key).replace(','+original_input_first_key,','+modified_input_first_key)
                selected_input2 = selected_input2.replace('_{'+original_input_first_key,'_{'+modified_input_first_key).replace(','+original_input_first_key,','+modified_input_first_key)
                # print(f'new_input2:{new_input2}, selected_input2:{selected_input2}')
                new_input_keys_list = ['bx', 'tx'] + new_input_keys_list[1:]
                new_input_values_list = [1] + new_input_values_list
                new_input_loop_type_list = ['B', 'B'] + new_input_loop_type_list[1:]
            new_input_loop_type_index_list=[idx for idx in range(len(new_input_loop_type_list)) if new_input_loop_type_list[idx]=='L']
            if len(new_input_loop_type_index_list)>0:
                has_transformation = True
                new_input_loop_type_list[new_input_loop_type_index_list[0]]='V'
                input_loop_notation = curate_loops(new_input_values_list, new_input_keys_list, new_input_loop_type_list)
                new_input_eq= input_loop_notation + '[' + new_input2 + '=' + selected_input2 + ';];'
                new_this_eq=this_raw_eq.replace(selected_input1, new_input1)
                transformed_part = new_input_eq + new_this_eq
                transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                transformed_IR_list.append(transformed_IR)
            # print(f'original IR:{row_equations_under_loops[op_index]}\ntransformed_part: {transformed_part}')
        # cache_write
        if len(candidate_outputs)>0:
            this_raw_eq= row_equations_under_loops[op_index]
            selected_output1 = random.choice(candidate_outputs)
            # print(f'selected_output: {selected_output1}')
            selected_output_memory_location = random.choice(['s', 'l'])
            output_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
            output_superscript = re.findall(r'\^\{.*?\}', selected_output1)[0]
            new_output_superscript = output_superscript.replace(',g}', ','+selected_output_memory_location+'}')
            output_full_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(selected_output1)
            output_full_subscript = output_full_subscript_list[0]
            new_output1= output_intermediate_names[0] + new_output_superscript + output_full_subscript
            output_subscript_list = re.findall(r'[a-zA-Z]+', output_full_subscript)
            output_loop_index_list = [idx for idx in range(len(keys_list[0])) if keys_list[0][idx] in output_subscript_list]
            new_output_keys_list = [keys_list[0][idx] for idx in output_loop_index_list]
            new_output_values_list = [values_list[0][idx] for idx in output_loop_index_list]
            new_output_loop_type_list = [loop_type_list[0][idx] for idx in output_loop_index_list]
            new_output2=new_output1
            selected_output2=selected_output1
            if 'bx' not in new_output_keys_list or 'tx' not in new_output_keys_list:
                original_output_first_key=new_output_keys_list[0]
                modified_output_first_key='bx*'+str(new_output_values_list[0])+'+tx'
                new_output2 = new_output2.replace('_{'+original_output_first_key,'_{'+modified_output_first_key).replace(','+original_output_first_key,','+modified_output_first_key)
                selected_output2=selected_output2.replace('_{'+original_output_first_key,'_{'+modified_output_first_key).replace(','+original_output_first_key,','+modified_output_first_key)
                # print(f'new_output2:{new_output2}, selected_output2:{selected_output2}')
                new_output_keys_list = ['bx', 'tx'] + new_output_keys_list[1:]
                new_output_values_list = [1] + new_output_values_list
                new_output_loop_type_list = ['B', 'B'] + new_output_loop_type_list[1:]
            new_output_loop_type_index_list= [idx for idx in range(len(new_output_loop_type_list)) if new_output_loop_type_list[idx]=='L']
            if len(new_output_loop_type_index_list)>0:
                has_transformation = True
                new_output_loop_type_list[new_output_loop_type_index_list[0]]='V'
                output_loop_notation = curate_loops(new_output_values_list, new_output_keys_list, new_output_loop_type_list)
                new_this_eq = this_raw_eq.replace(selected_output1, new_output1)
                new_output_eq = output_loop_notation + '[' + selected_output2 + '=' + new_output2 + ';];'
                transformed_part = new_this_eq + new_output_eq
                transformed_IR = ''.join(row_equations_under_loops[:op_index]) + transformed_part + ''.join(row_equations_under_loops[op_index+1:])
                transformed_IR_list.append(transformed_IR)
            # print(f'original IR:{row_equations_under_loops[op_index]}\ntransformed_part: {transformed_part}')
    return [IR], transformed_IR_list, has_transformation

def apply_set_storage_scope_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    set_storage_scope_index_mapping = judge_set_storage_scope_condition(input_output_name, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    for intermediate_variable in set_storage_scope_index_mapping.keys():
        op_index_list=set_storage_scope_index_mapping[intermediate_variable]
        # print(f'intermediate_variable: {intermediate_variable}, op_index_list: {op_index_list}')
        selected_input_memory_location= random.choice(['s', 'l'])
        transform_row_equations_under_loops = row_equations_under_loops.copy()
        for op_index in op_index_list:
            this_raw_eq = row_equations_under_loops[op_index]
            this_input_output_list=list(set(eq_outputs_under_loops[op_index]+eq_inputs_under_loops[op_index]))
            full_intermediate_variable=''
            for this_input_output in this_input_output_list:
                if intermediate_variable+'^{' in this_input_output:
                    full_intermediate_variable=this_input_output
                    break
            if full_intermediate_variable!='':
                superscript = re.findall(r'\^\{.*?\}', full_intermediate_variable)[0]
                new_superscript = superscript.replace(',g}', ','+selected_input_memory_location+'}')
                new_this_raw_eq = this_raw_eq.replace(intermediate_variable+superscript, intermediate_variable+new_superscript)
                transform_row_equations_under_loops[op_index] = new_this_raw_eq
                has_transformation = True
        transformed_IR = ''.join(transform_row_equations_under_loops)
        transformed_IR_list.append(transformed_IR)
        original_IR_list.append(IR)
        # print(f'original IR: {row_equations_under_loops}\ntransformed IR: {transformed_IR}')
    return original_IR_list, transformed_IR_list, has_transformation

def apply_set_storage_layout_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    #transformation types: 
    # 1.transpose(multiple dims(>1)->multiple dims, same number of dim) 
    # 2. reshape(multiple dims/single dim(>=1, factorization)->more dims, multiple dims (>1)->less dims)
    # 3. flatten (multiple dims(>1)->single dim)
    # 4. squeeze (multiple dims including 1(>1)-> less dims)
    candidate_inputs_transpose_flatten_reshape2_mapping, candidate_inputs_reshape1_mapping, candidate_inputs_squeeze_mapping = judge_set_storage_layout_condition(input_output_name, loops,simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    for simplified_input in candidate_inputs_transpose_flatten_reshape2_mapping.keys():
        has_transformation = True
        candidate_transpose_flatten_reshape2_values= candidate_inputs_transpose_flatten_reshape2_mapping[simplified_input]
        transpose_row_equations_under_loops = row_equations_under_loops.copy()
        flatten_row_equations_under_loops = row_equations_under_loops.copy()
        reshape2_row_equations_under_loops = row_equations_under_loops.copy()
        #transpose
        temp_subscript_details=candidate_transpose_flatten_reshape2_values[0][2]
        shuffled_subscript_details_idx = random_shuffule(list(range(len(temp_subscript_details))))
        #reshape2
        temp_reshape2_selected_axis=random.sample(temp_subscript_details, 2)
        reshape2_selected_axis_index = [temp_subscript_details.index(item) for item in temp_reshape2_selected_axis]
        if len(temp_subscript_details)>=3:
            shuffled_reshape2_subscript_idx_list = random_shuffule(list(range(len(temp_subscript_details)-1)))
        else:
            shuffled_reshape2_subscript_idx_list = [0]
        for candidate_transpose_flatten_reshape2_value_idx in range(len(candidate_transpose_flatten_reshape2_values)):
            op_index, this_input, subscript_details=candidate_transpose_flatten_reshape2_values[candidate_transpose_flatten_reshape2_value_idx]
            if candidate_transpose_flatten_reshape2_value_idx==0:
                values_list, keys_list, _, _ = split_loops_into_value_and_index([loops[op_index]])
                key_value_mapping={keys_list[0][idx]: values_list[0][idx]-1 for idx in range(len(keys_list[0]))}
                subscript_details_value =[eval(expr, {}, key_value_mapping)+1 for expr in subscript_details]
            #transpose
            this_input_superscript= re.findall(r'\^{[a-zA-Z0-9,]*}', this_input)[0]
            shuffled_subscript_details = [subscript_details[idx] for idx in shuffled_subscript_details_idx]
            shuffled_subscript_in_eq = '_{' + ','.join(shuffled_subscript_details) + '}'
            new_transpose_input_in_eq = simplified_input+ this_input_superscript + shuffled_subscript_in_eq
            transpose_row_equations_under_loops[op_index] = transpose_row_equations_under_loops[op_index].replace(this_input, new_transpose_input_in_eq)
            #flatten
            flatten_subscript_details_in_eq=''
            len_subscript=len(subscript_details)
            reverse_subscript_details=subscript_details[::-1]
            reverse_subscript_details_value=subscript_details_value[::-1]
            for idx in range(len_subscript-1):
                if math.prod(reverse_subscript_details_value[idx+1:])>1:
                    if len(re.sub('[a-z]+','', reverse_subscript_details[idx]))==0:
                        flatten_subscript_details_in_eq+=reverse_subscript_details[idx]+'*'+str(math.prod(reverse_subscript_details_value[idx+1:]))+'+'
                    elif not reverse_subscript_details[idx].isdigit():
                        flatten_subscript_details_in_eq+='('+reverse_subscript_details[idx]+')*'+str(math.prod(reverse_subscript_details_value[idx+1:]))+'+'
                else:
                    if len(re.sub('[a-z]+','', reverse_subscript_details[idx]))==0:
                        flatten_subscript_details_in_eq+=reverse_subscript_details[idx]+'+'
                    elif not reverse_subscript_details[idx].isdigit():
                        flatten_subscript_details_in_eq+='('+reverse_subscript_details[idx]+')+'
            if len(re.sub('[a-z]+','', reverse_subscript_details[len_subscript-1]))==0:
                flatten_subscript_details_in_eq+=reverse_subscript_details[len_subscript-1] 
            elif not reverse_subscript_details[len_subscript-1].isdigit():
                flatten_subscript_details_in_eq+='('+reverse_subscript_details[len_subscript-1]+')'
            else:
                flatten_subscript_details_in_eq=flatten_subscript_details_in_eq[-1]
            flatten_input_in_eq = simplified_input + this_input_superscript + '_{' + flatten_subscript_details_in_eq+ '}'
            flatten_row_equations_under_loops[op_index] = flatten_row_equations_under_loops[op_index].replace(this_input, flatten_input_in_eq)
            #reshape2
            reshape2_selected_axis_index.sort()
            reshape2_selected_axis_index=reshape2_selected_axis_index[::-1]
            reshape2_selected_axis=[subscript_details[idx] for idx in reshape2_selected_axis_index]
            reshape2_selected_axis_value=[subscript_details_value[idx] for idx in reshape2_selected_axis_index]
            left_reshape2_subscript_list=[item for item in subscript_details if item not in reshape2_selected_axis]#list(set(subscript_details)-set(reshape2_selected_axis))
            fused_reshape2_axis=''
            if  len(re.sub('[a-z]+','', reshape2_selected_axis[0]))==0:
                fused_reshape2_axis+=reshape2_selected_axis[0]+'*'+str(reshape2_selected_axis_value[1])+'+'  
            elif not reshape2_selected_axis[0].isdigit():
                fused_reshape2_axis+='('+reshape2_selected_axis[0]+')*'+str(reshape2_selected_axis_value[1])+'+'
            if len(re.sub('[a-z]+','', reshape2_selected_axis[1]))==0:
                fused_reshape2_axis+=reshape2_selected_axis[1]
            elif not reshape2_selected_axis[1].isdigit():
                fused_reshape2_axis+='('+reshape2_selected_axis[1]+')'
            temp_reshape2_subscript_list = left_reshape2_subscript_list + [fused_reshape2_axis]
            reshape2_subscript_list_in_eq = [temp_reshape2_subscript_list[idx] for idx in shuffled_reshape2_subscript_idx_list]
            # print(f'temp_reshape2_subscript_list:{temp_reshape2_subscript_list}, shuffled_reshape2_subscript_idx_list:{shuffled_reshape2_subscript_idx_list}, reshape2_subscript_list_in_eq:{reshape2_subscript_list_in_eq}')
            reshape2_input_in_eq = simplified_input + this_input_superscript + '_{' + ','.join(reshape2_subscript_list_in_eq) + '}'
            reshape2_row_equations_under_loops[op_index] = reshape2_row_equations_under_loops[op_index].replace(this_input, reshape2_input_in_eq)
        transpose_IR=''.join(transpose_row_equations_under_loops)
        print(f'simplified_input:{simplified_input}, transpose_IR: {transpose_IR}')
        flatten_IR=''.join(flatten_row_equations_under_loops)
        print(f'simplified_input:{simplified_input}, flatten_IR: {flatten_IR}')
        reshape2_IR=''.join(reshape2_row_equations_under_loops)
        print(f'simplified_input:{simplified_input}, reshape2_IR: {reshape2_IR}')
        transformed_IR_list.append(transpose_IR)
        transformed_IR_list.append(flatten_IR)
        transformed_IR_list.append(reshape2_IR)
        original_IR_list.append(IR)
        original_IR_list.append(IR)
        original_IR_list.append(IR)
    for simplified_input in candidate_inputs_reshape1_mapping.keys():
        has_transformation = True
        candidate_reshape1_values = candidate_inputs_reshape1_mapping[simplified_input]
        reshape1_row_equations_under_loops = row_equations_under_loops.copy()
        temp_values=candidate_reshape1_values[0][5]
        subscript_idx=candidate_reshape1_values[0][4]
        reshape1_factorization_list=random.choice(temp_values)
        # print(f'simplified_input:{simplified_input},value:{candidate_reshape1_values}')
        for candidate_reshape1_value_idx in range(len(candidate_reshape1_values)):
            op_index, this_input, subscript_details, subscript_details_value, _, _ = candidate_reshape1_values[candidate_reshape1_value_idx]
            this_input_superscript= re.findall(r'\^{[a-zA-Z0-9,]*}', this_input)[0]
            #reshape1
            if candidate_reshape1_value_idx==0:
                values_list, keys_list, _, _ = split_loops_into_value_and_index([loops[op_index]])
                key_value_mapping={keys_list[0][idx]: values_list[0][idx]-1 for idx in range(len(keys_list[0]))}
            if re.sub(r'[a-z]+','',subscript_details[subscript_idx])!='':
                selected_subscript=['('+subscript_details[subscript_idx]+')//'+str(reshape1_factorization_list[1]), '('+subscript_details[subscript_idx]+')%'+str(reshape1_factorization_list[1])]
            else:
                selected_subscript=[subscript_details[subscript_idx]+'//'+str(reshape1_factorization_list[1]), subscript_details[subscript_idx]+'%'+str(reshape1_factorization_list[1])]
            subscript_details_in_eq=subscript_details[:subscript_idx]+selected_subscript+ subscript_details[subscript_idx+1:]
            # print(f'subscript_details_in_eq:{subscript_details_in_eq}')
            reshape1_input_in_eq = simplified_input + this_input_superscript + '_{' + ','.join(subscript_details_in_eq) + '}'
            reshape1_row_equations_under_loops[op_index] = reshape1_row_equations_under_loops[op_index].replace(this_input, reshape1_input_in_eq)
        reshape1_IR= ''.join(reshape1_row_equations_under_loops)
        # print(f'this_input:{this_input}, reshape1_IR: {reshape1_IR}')
        transformed_IR_list.append(reshape1_IR)
        original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_set_storage_align_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    set_storage_align_index_mapping = judge_set_storage_align_condition(loops, simplified_eqs_under_loops, input_output_name, eq_outputs_under_loops, simplified_eq_outputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    for simplified_input_output in set_storage_align_index_mapping.keys():
        has_transformation = True
        # print(f'simplified_input_output: {simplified_input_output}')
        op_index, subscript_details_value= set_storage_align_index_mapping[simplified_input_output]
        selcted_axis = random.choice(list(range(len(subscript_details_value))))
        factor = random.choice([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
        offset = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        align_eq=simplified_input_output+'=align('+str(selcted_axis)+','+str(factor)+','+str(offset)+');'#'D['+simplified_input_output+'=align('+str(selcted_axis)+','+str(factor)+','+str(offset)+');];'
        # print(f'align_eq: {align_eq}')
        # transformed_IR= align_eq + ''.join(row_equations_under_loops)
        # transformed_IR_list.append(transformed_IR)
        new_this_eq=loops[op_index]+'['+align_eq+';'.join(equations_under_loops[op_index])+';];'
        # print(f'new_this_eq: {new_this_eq}')
        transformed_IR = ''.join(row_equations_under_loops[:op_index]) + new_this_eq + ''.join(row_equations_under_loops[op_index+1:])
        transformed_IR_list.append(transformed_IR)
    return [IR], transformed_IR_list, has_transformation

def apply_asynchronous_pipeline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops, notation='^{ap_0}'):
    #asynchronous pipeline
    asynchronous_pipeline_index_list = judge_asynchronous_pipeline_condition(simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    for index in range(len(asynchronous_pipeline_index_list)):
        op_index, candidate_inputs=asynchronous_pipeline_index_list[index]
        this_loop = loops[op_index]
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([this_loop])
        for this_input in candidate_inputs:
            has_transformation = True
            this_eq=';'.join(equations_under_loops[op_index])
            this_input_superscript = re.findall(r'\^{[a-zA-Z0-9,]*}', this_input)[0]
            memory_location = random.choice(['s', 'l'])
            new_input_superscript = this_input_superscript.replace(',g}', ','+memory_location+'}')
            this_input_subscript_list,this_simplified_input=find_subscripts_of_input_output_and_simplified_version(this_input)
            new_eq=this_eq.replace(this_simplified_input+this_input_superscript,this_simplified_input+new_input_superscript)
            new_input=this_input.replace(this_simplified_input+this_input_superscript,this_simplified_input+new_input_superscript)
            input_subscript_details=generate_subscript_details(this_input_subscript_list)
            input_var_subscript=[item for item in input_subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))>0]
            remove_var_subscript_details=[item for item in input_subscript_details if len(re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item))==0]
            if len(input_var_subscript)>0:
                input_var_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(input_var_subscript[0])
                input_var_subscript_details=generate_subscript_details(input_var_subscript_list)
            else:
                input_var_subscript_details=[]
            subscript_details=remove_var_subscript_details+input_var_subscript_details
            subscript_keys=[]
            for subscript_detail in subscript_details:
                subscript_keys.extend(re.findall(r'[a-zA-Z]+', subscript_detail))
            subscript_keys=list(set(subscript_keys))
            # print(f'this_input:{this_input}\nsubscript_details:{subscript_details}\nsubscript_keys:{subscript_keys}')
            keys_index_list=[idx for idx in range(len(keys_list[0])) if keys_list[0][idx] in subscript_keys]
            orther_keys_index_list=[idx for idx in range(len(keys_list[0])) if keys_list[0][idx] not in subscript_keys]
            in_keys_list= [keys_list[0][idx] for idx in keys_index_list]
            in_values_list= [values_list[0][idx] for idx in keys_index_list]
            in_loop_type_list= [loop_type_list[0][idx] for idx in keys_index_list]
            out_keys_list= [keys_list[0][idx] for idx in orther_keys_index_list]
            out_values_list= [values_list[0][idx] for idx in orther_keys_index_list]
            out_loop_type_list= [loop_type_list[0][idx] for idx in orther_keys_index_list]
            in_loop_notations=curate_loops(in_values_list, in_keys_list, in_loop_type_list)
            out_loop_notations=curate_loops(out_values_list, out_keys_list, out_loop_type_list)
            new_raw_eq=out_loop_notations+'['+notation+in_loop_notations+'['+new_input+'='+this_input+';];'+in_loop_notations+'['+new_eq+';];]'+notation+';'
            transformed_IR = ''.join(row_equations_under_loops[:op_index]) + new_raw_eq + ''.join(row_equations_under_loops[op_index+1:])
            transformed_IR_list.append(transformed_IR)
            # print(f'new_raw_eq: {new_raw_eq}')
    return [IR], transformed_IR_list, has_transformation

def apply_double_buffer_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    #double buffer
    original_IR, transformed_IR_list, has_transformation=apply_asynchronous_pipeline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops, notation='^{bd_0}')
    return original_IR, transformed_IR_list, has_transformation

def apply_precompute_indices_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    precompute_indices_index_list = judge_precompute_indices_condition(simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    for index in range(len(precompute_indices_index_list)):
        has_transformation=True
        op_index, remove_num_var_alpha_subscript_details= precompute_indices_index_list[index]
        values_list, keys_list, loop_type_list, _ = split_loops_into_value_and_index([loops[op_index]])
        for subscript in remove_num_var_alpha_subscript_details:
            subscript_keys=re.findall(r'[a-z]+',subscript)
            subscript_key_index_list=[idx for idx in range(len(keys_list[0])) if keys_list[0][idx] in subscript_keys]
            intermediate_names, name_start_idx = generate_names(1, name_start_idx)
            intermediate_superscript='^{i64,g}'
            intermediate_subscript_in_second_eq='_{'+','.join(subscript_keys)+'}'
            # print(f'subscript:{subscript},subscript_keys:{subscript_keys},subscript_key_index_list:{subscript_key_index_list}')
            new_values_list=[values_list[0][idx] for idx in subscript_key_index_list]
            new_keys_list=[keys_list[0][idx] for idx in subscript_key_index_list]
            new_loop_type_list=[loop_type_list[0][idx] for idx in subscript_key_index_list]
            intermediate_subscript_in_first_eq=intermediate_subscript_in_second_eq
            subscript_in_first_eq=subscript
            has_tx=True
            # print(f'new_values_list:{new_values_list},new_keys_list:{new_keys_list},new_loop_type_list:{new_loop_type_list}')
            # print(f'intermediate_subscript_in_first_eq:{intermediate_subscript_in_first_eq},subscript_in_first_eq:{subscript_in_first_eq}')
            if 'tx' not in intermediate_subscript_in_first_eq:
                can_be_selected_axis=[idx for idx in range(len(new_values_list)) if new_values_list[idx]<1024]
                if len(can_be_selected_axis)>0:
                    has_tx=True
                    axis=can_be_selected_axis[0]
                    original_output_first_key=new_keys_list[axis]
                    modified_output_first_key='tx'
                    split_subscript_in_first_eq_by_subscript = ['_{'+subscript_in_first_eq+'}']
                    split_intermediate_subscript_in_first_eq_by_subscript = [intermediate_subscript_in_first_eq]
                    replaced_split_subscript_in_first_eq_var_list_eqs=replace_subscript_in_split_eqs(split_subscript_in_first_eq_by_subscript, original_output_first_key, modified_output_first_key)
                    replaced_split_intermediate_subscript_in_first_eq_var_list_eqs=replace_subscript_in_split_eqs(split_intermediate_subscript_in_first_eq_by_subscript, original_output_first_key, modified_output_first_key)
                    subscript_in_first_eq=replaced_split_subscript_in_first_eq_var_list_eqs[0].replace('_{','').replace('}','')
                    intermediate_subscript_in_first_eq=replaced_split_intermediate_subscript_in_first_eq_var_list_eqs[0]
                    # print(f'intermediate_subscript_in_first_eq:{intermediate_subscript_in_first_eq},subscript_in_first_eq:{subscript_in_first_eq}')
                    new_keys_list = new_keys_list[:axis] + ['tx'] + new_keys_list[axis+1:]
                    new_loop_type_list = new_loop_type_list[:axis] + ['B'] + new_loop_type_list[axis+1:]
                else:
                    has_tx=False
            if has_tx:
                new_loop_notation = curate_loops(new_values_list, new_keys_list, new_loop_type_list)
                new_first_eq=new_loop_notation+'['+intermediate_names[0]+intermediate_superscript+intermediate_subscript_in_first_eq+'='+subscript_in_first_eq+';];'
                intermediate_var=intermediate_names[0]+intermediate_superscript+intermediate_subscript_in_second_eq
                # print(f'equations_under_loops[op_index]:{equations_under_loops[op_index]}\nsubscript:{subscript}, intermediate_var:{intermediate_var}')
                split_this_new_eq_by_subscript = split_eq_by_subscript(';'.join(equations_under_loops[op_index]))
                replaced_split_this_new_eq=replace_comb_subscript_in_split_eqs(split_this_new_eq_by_subscript, subscript, intermediate_var)
                this_new_eq=''.join(replaced_split_this_new_eq)
                # print(f'split_this_new_eq_by_subscript:{split_this_new_eq_by_subscript}\nreplaced_split_this_new_eq:{replaced_split_this_new_eq}\nthis_new_eq:{this_new_eq}')
                # this_new_eq=';'.join(equations_under_loops[op_index]).replace('{'+subscript, '{'+intermediate_var).replace('{'+subscript, '{'+intermediate_var)
                new_second_eq=loops[op_index]+'['+this_new_eq+';];'
                # print(f'new_first_eq:{new_first_eq}\nnew_second_eq: {new_second_eq}')
                transfromed_IR=''.join(row_equations_under_loops[:op_index]) + new_first_eq + new_second_eq + ''.join(row_equations_under_loops[op_index+1:])
                transformed_IR_list.append(transfromed_IR)
                original_IR_list.append(IR)
            name_start_idx=name_start_idx-1
    return original_IR_list, transformed_IR_list, has_transformation

def apply_sympy_to_IR(IR, factorization_index_list,has_transformation, original_IR_list, transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops):
    for index in range(len(factorization_index_list)):
        op_index, eq_index, new_expr = factorization_index_list[index]
        this_eqs = equations_under_loops[op_index][eq_index]
        this_output=this_eqs.split('=')[0] 
        if 'tx' in this_output:
            # print(f'new_expr:{new_expr}')
            # print(f'op_index: {op_index}, eq_index: {eq_index}, old_expr:{equations_under_loops[op_index][eq_index]}, new_expr: {new_expr}')
            expr, _=transform_from_original_simplified_expr_to_original_expr(new_expr, simplified_eq_inputs_under_loops[op_index], eq_inputs_under_loops[op_index])
            
            new_eq=this_output + '=' + expr + ';'
            new_eqs=';'.join(equations_under_loops[op_index][:eq_index])
            if new_eqs!='':
                new_eqs = new_eqs+';'+new_eq+ ''.join(equations_under_loops[op_index][eq_index+1:])
            else:
                new_eqs = new_eq + ''.join(equations_under_loops[op_index][eq_index+1:])
            if new_eqs[-1]==';':
                transformed_IR = ''.join(row_equations_under_loops[:op_index]) + loops[op_index]+'['+new_eqs +'];'+ ''.join(row_equations_under_loops[op_index+1:])
            else:
                transformed_IR = ''.join(row_equations_under_loops[:op_index]) + loops[op_index]+'['+new_eqs +';];'+ ''.join(row_equations_under_loops[op_index+1:])
            # print(f'transformed_IR: {transformed_IR}')
            if transformed_IR not in transformed_IR_list:
                has_transformation = True
                transformed_IR_list.append(transformed_IR)
                original_IR_list.append(IR)
    return has_transformation, original_IR_list, transformed_IR_list

def apply_factorization_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    factorization_index_list = judge_factorization_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    has_transformation,original_IR_list, transformed_IR_list=apply_sympy_to_IR(IR, factorization_index_list,has_transformation, original_IR_list, transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    _,one_step_transformed_IR_list,_=apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    # print(f'one_step_transformed_IR_list:{one_step_transformed_IR_list}')
    for one_step_transformed_IR_idx in range(len(one_step_transformed_IR_list)):
        one_step_transformed_IR=one_step_transformed_IR_list[one_step_transformed_IR_idx]
        # print(f'one_step_transformed_IR: {one_step_transformed_IR}')
        one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops,one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops = split_IR_to_equations(one_step_transformed_IR)
        # print(f'one_step_row_equations_under_loops:{one_step_row_equations_under_loops},one_step_equations_under_loops:{one_step_equations_under_loops}\none_step_eq_inputs_under_loops:{one_step_eq_inputs_under_loops}')
        _,two_step_transformed_IR_list,_=apply_compute_inline_to_IR(one_step_transformed_IR, input_output_name, name_start_idx, one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops, one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        # print(f'two_step_transformed_IR_list:{two_step_transformed_IR_list}')
        for two_step_transformed_IR_idx in range(len(two_step_transformed_IR_list)):
            two_step_transformed_IR=two_step_transformed_IR_list[two_step_transformed_IR_idx]
            # print(f'two_step_transformed_IR: {two_step_transformed_IR}')
            two_step_row_equations_under_loops, two_step_loops, two_step_equations_under_loops, two_step_eq_outputs_under_loops,two_step_eq_inputs_under_loops, two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops = split_IR_to_equations(two_step_transformed_IR)
            two_step_factorization_index_list = judge_factorization_condition(two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops)
            has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(two_step_transformed_IR, two_step_factorization_index_list,has_transformation,original_IR_list, transformed_IR_list, two_step_loops, two_step_row_equations_under_loops, two_step_equations_under_loops, two_step_simplified_eq_inputs_under_loops, two_step_eq_inputs_under_loops)
        one_step_factorization_index_list = judge_factorization_condition(one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(one_step_transformed_IR, one_step_factorization_index_list, has_transformation,original_IR_list, transformed_IR_list, one_step_loops, one_step_row_equations_under_loops, one_step_equations_under_loops, one_step_simplified_eq_inputs_under_loops, one_step_eq_inputs_under_loops)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_expand_factorization_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    expand_factorization_index_list = judge_expand_factorization_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    has_transformation,original_IR_list, transformed_IR_list=apply_sympy_to_IR(IR, expand_factorization_index_list,has_transformation, original_IR_list, transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    _,one_step_transformed_IR_list,_=apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    # print(f'one_step_transformed_IR_list:{one_step_transformed_IR_list}')
    for one_step_transformed_IR_idx in range(len(one_step_transformed_IR_list)):
        one_step_transformed_IR=one_step_transformed_IR_list[one_step_transformed_IR_idx]
        # print(f'one_step_transformed_IR: {one_step_transformed_IR}')
        one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops,one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops = split_IR_to_equations(one_step_transformed_IR)
        # print(f'one_step_row_equations_under_loops:{one_step_row_equations_under_loops},one_step_equations_under_loops:{one_step_equations_under_loops}\none_step_eq_inputs_under_loops:{one_step_eq_inputs_under_loops}')
        _,two_step_transformed_IR_list,_=apply_compute_inline_to_IR(one_step_transformed_IR, input_output_name, name_start_idx, one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops, one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        # print(f'two_step_transformed_IR_list:{two_step_transformed_IR_list}')
        for two_step_transformed_IR_idx in range(len(two_step_transformed_IR_list)):
            two_step_transformed_IR=two_step_transformed_IR_list[two_step_transformed_IR_idx]
            # print(f'two_step_transformed_IR: {two_step_transformed_IR}')
            two_step_row_equations_under_loops, two_step_loops, two_step_equations_under_loops, two_step_eq_outputs_under_loops,two_step_eq_inputs_under_loops, two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops = split_IR_to_equations(two_step_transformed_IR)
            two_step_expand_factorization_index_list = judge_expand_factorization_condition(two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops)
            has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(two_step_transformed_IR, two_step_expand_factorization_index_list,has_transformation,original_IR_list, transformed_IR_list, two_step_loops, two_step_row_equations_under_loops, two_step_equations_under_loops, two_step_simplified_eq_inputs_under_loops, two_step_eq_inputs_under_loops)
        one_step_expand_factorization_index_list = judge_expand_factorization_condition(one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(one_step_transformed_IR, one_step_expand_factorization_index_list, has_transformation,original_IR_list, transformed_IR_list, one_step_loops, one_step_row_equations_under_loops, one_step_equations_under_loops, one_step_simplified_eq_inputs_under_loops, one_step_eq_inputs_under_loops)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_cancellation_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    cancellation_index_list = judge_cancellation_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(IR, cancellation_index_list,has_transformation, original_IR_list, transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    _,one_step_transformed_IR_list,_=apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    # print(f'one_step_transformed_IR_list:{one_step_transformed_IR_list}')
    for one_step_transformed_IR_idx in range(len(one_step_transformed_IR_list)):
        one_step_transformed_IR=one_step_transformed_IR_list[one_step_transformed_IR_idx]
        # print(f'one_step_transformed_IR: {one_step_transformed_IR}')
        one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops,one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops = split_IR_to_equations(one_step_transformed_IR)
        # print(f'one_step_row_equations_under_loops:{one_step_row_equations_under_loops},one_step_equations_under_loops:{one_step_equations_under_loops}\none_step_eq_inputs_under_loops:{one_step_eq_inputs_under_loops}')
        _,two_step_transformed_IR_list,_=apply_compute_inline_to_IR(one_step_transformed_IR, input_output_name, name_start_idx, one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops, one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        # print(f'two_step_transformed_IR_list:{two_step_transformed_IR_list}')
        for two_step_transformed_IR_idx in range(len(two_step_transformed_IR_list)):
            two_step_transformed_IR=two_step_transformed_IR_list[two_step_transformed_IR_idx]
            # print(f'two_step_transformed_IR: {two_step_transformed_IR}')
            two_step_row_equations_under_loops, two_step_loops, two_step_equations_under_loops, two_step_eq_outputs_under_loops,two_step_eq_inputs_under_loops, two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops = split_IR_to_equations(two_step_transformed_IR)
            two_step_cancellation_index_list = judge_cancellation_condition(two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops)
            has_transformation,original_IR_list, transformed_IR_list=apply_sympy_to_IR(two_step_transformed_IR, two_step_cancellation_index_list,has_transformation,original_IR_list, transformed_IR_list, two_step_loops, two_step_row_equations_under_loops, two_step_equations_under_loops, two_step_simplified_eq_inputs_under_loops, two_step_eq_inputs_under_loops)
        one_step_cancellation_index_list = judge_cancellation_condition(one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(one_step_transformed_IR, one_step_cancellation_index_list, has_transformation, original_IR_list, transformed_IR_list, one_step_loops, one_step_row_equations_under_loops, one_step_equations_under_loops, one_step_simplified_eq_inputs_under_loops, one_step_eq_inputs_under_loops)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_expand_cancellation_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    original_IR, transformed_IR_list, has_transformation=apply_cancellation_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    return transformed_IR_list,original_IR, has_transformation

# def apply_apart_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
#     apart_index_list = judge_apart_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
#     has_transformation = False
#     transformed_IR_list = []
#     original_IR_list=[]
#     has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(IR, apart_index_list,has_transformation, original_IR_list, transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
#     _,one_step_transformed_IR_list,_=apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
#     # print(f'one_step_transformed_IR_list:{one_step_transformed_IR_list}')
#     for one_step_transformed_IR_idx in range(len(one_step_transformed_IR_list)):
#         one_step_transformed_IR=one_step_transformed_IR_list[one_step_transformed_IR_idx]
#         # print(f'one_step_transformed_IR: {one_step_transformed_IR}')
#         one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops,one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops = split_IR_to_equations(one_step_transformed_IR)
#         # print(f'one_step_row_equations_under_loops:{one_step_row_equations_under_loops},one_step_equations_under_loops:{one_step_equations_under_loops}\none_step_eq_inputs_under_loops:{one_step_eq_inputs_under_loops}')
#         _,two_step_transformed_IR_list,_=apply_compute_inline_to_IR(one_step_transformed_IR, input_output_name, name_start_idx, one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops, one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
#         # print(f'two_step_transformed_IR_list:{two_step_transformed_IR_list}')
#         for two_step_transformed_IR_idx in range(len(two_step_transformed_IR_list)):
#             two_step_transformed_IR=two_step_transformed_IR_list[two_step_transformed_IR_idx]
#             # print(f'two_step_transformed_IR: {two_step_transformed_IR}')
#             two_step_row_equations_under_loops, two_step_loops, two_step_equations_under_loops, two_step_eq_outputs_under_loops,two_step_eq_inputs_under_loops, two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops = split_IR_to_equations(two_step_transformed_IR)
#             two_step_apart_index_list = judge_apart_condition(two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops)
#             has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(two_step_transformed_IR, two_step_apart_index_list,has_transformation,original_IR_list, transformed_IR_list, two_step_loops, two_step_row_equations_under_loops, two_step_equations_under_loops, two_step_simplified_eq_inputs_under_loops, two_step_eq_inputs_under_loops)
#         one_step_apart_index_list = judge_apart_condition(one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
#         has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(one_step_transformed_IR, one_step_apart_index_list, has_transformation, original_IR_list, transformed_IR_list, one_step_loops, one_step_row_equations_under_loops, one_step_equations_under_loops, one_step_simplified_eq_inputs_under_loops, one_step_eq_inputs_under_loops)
#     return original_IR_list, transformed_IR_list, has_transformation

def apply_together_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    together_index_list = judge_together_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    has_transformation,original_IR_list, transformed_IR_list=apply_sympy_to_IR(IR, together_index_list,has_transformation, original_IR_list,transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    _,one_step_transformed_IR_list,_=apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    # print(f'one_step_transformed_IR_list:{one_step_transformed_IR_list}')
    for one_step_transformed_IR_idx in range(len(one_step_transformed_IR_list)):
        one_step_transformed_IR=one_step_transformed_IR_list[one_step_transformed_IR_idx]
        # print(f'one_step_transformed_IR: {one_step_transformed_IR}')
        one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops,one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops = split_IR_to_equations(one_step_transformed_IR)
        # print(f'one_step_row_equations_under_loops:{one_step_row_equations_under_loops},one_step_equations_under_loops:{one_step_equations_under_loops}\none_step_eq_inputs_under_loops:{one_step_eq_inputs_under_loops}')
        _,two_step_transformed_IR_list,_=apply_compute_inline_to_IR(one_step_transformed_IR, input_output_name, name_start_idx, one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops, one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        # print(f'two_step_transformed_IR_list:{two_step_transformed_IR_list}')
        for two_step_transformed_IR_idx in range(len(two_step_transformed_IR_list)):
            two_step_transformed_IR=two_step_transformed_IR_list[two_step_transformed_IR_idx]
            # print(f'two_step_transformed_IR: {two_step_transformed_IR}')
            two_step_row_equations_under_loops, two_step_loops, two_step_equations_under_loops, two_step_eq_outputs_under_loops,two_step_eq_inputs_under_loops, two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops = split_IR_to_equations(two_step_transformed_IR)
            two_step_together_index_list = judge_together_condition(two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops)
            has_transformation,original_IR_list, transformed_IR_list=apply_sympy_to_IR(two_step_transformed_IR, two_step_together_index_list,has_transformation,original_IR_list,transformed_IR_list, two_step_loops, two_step_row_equations_under_loops, two_step_equations_under_loops, two_step_simplified_eq_inputs_under_loops, two_step_eq_inputs_under_loops)
        one_step_together_index_list = judge_together_condition(one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        has_transformation, original_IR_list,transformed_IR_list=apply_sympy_to_IR(one_step_transformed_IR, one_step_together_index_list, has_transformation, original_IR_list,transformed_IR_list, one_step_loops, one_step_row_equations_under_loops, one_step_equations_under_loops, one_step_simplified_eq_inputs_under_loops, one_step_eq_inputs_under_loops)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_apart_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    original_IR, transformed_IR_list, has_transformation=apply_together_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    return transformed_IR_list,original_IR, has_transformation

def apply_trig_expand_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    trig_expand_index_list = judge_trig_expand_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list = []
    two_step_trig_expand_original_IR_list = []
    three_step_trig_expand_original_IR_list = []
    two_step_trig_expand_transformed_IR_list=[]
    three_step_trig_expand_transformed_IR_list = []
    has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(IR, trig_expand_index_list,has_transformation, original_IR_list, transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    _,one_step_transformed_IR_list,_=apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    # print(f'one_step_transformed_IR_list:{one_step_transformed_IR_list}')
    for one_step_transformed_IR_idx in range(len(one_step_transformed_IR_list)):
        one_step_transformed_IR=one_step_transformed_IR_list[one_step_transformed_IR_idx]
        # print(f'one_step_transformed_IR: {one_step_transformed_IR}')
        one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops,one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops = split_IR_to_equations(one_step_transformed_IR)
        # print(f'one_step_row_equations_under_loops:{one_step_row_equations_under_loops},one_step_equations_under_loops:{one_step_equations_under_loops}\none_step_eq_inputs_under_loops:{one_step_eq_inputs_under_loops}')
        _,two_step_transformed_IR_list,_=apply_compute_inline_to_IR(one_step_transformed_IR, input_output_name, name_start_idx, one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops, one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        # print(f'two_step_transformed_IR_list:{two_step_transformed_IR_list}')
        for two_step_transformed_IR_idx in range(len(two_step_transformed_IR_list)):
            two_step_transformed_IR=two_step_transformed_IR_list[two_step_transformed_IR_idx]
            # print(f'two_step_transformed_IR: {two_step_transformed_IR}')
            two_step_row_equations_under_loops, two_step_loops, two_step_equations_under_loops, two_step_eq_outputs_under_loops,two_step_eq_inputs_under_loops, two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops = split_IR_to_equations(two_step_transformed_IR)
            two_step_trig_expand_index_list = judge_trig_expand_condition(two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops)
            has_transformation, three_step_trig_expand_original_IR_list, three_step_trig_expand_transformed_IR_list=apply_sympy_to_IR(two_step_transformed_IR, two_step_trig_expand_index_list,has_transformation,three_step_trig_expand_original_IR_list, three_step_trig_expand_transformed_IR_list, two_step_loops, two_step_row_equations_under_loops, two_step_equations_under_loops, two_step_simplified_eq_inputs_under_loops, two_step_eq_inputs_under_loops)
        one_step_trig_expand_index_list = judge_trig_expand_condition(one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        has_transformation, two_step_trig_expand_original_IR_list, two_step_trig_expand_transformed_IR_list=apply_sympy_to_IR(one_step_transformed_IR, one_step_trig_expand_index_list, has_transformation, two_step_trig_expand_original_IR_list, two_step_trig_expand_transformed_IR_list, one_step_loops, one_step_row_equations_under_loops, one_step_equations_under_loops, one_step_simplified_eq_inputs_under_loops, one_step_eq_inputs_under_loops)
    return original_IR_list + two_step_trig_expand_original_IR_list + three_step_trig_expand_original_IR_list, list(set(transformed_IR_list+two_step_trig_expand_transformed_IR_list+three_step_trig_expand_transformed_IR_list)), has_transformation

def apply_powsimp_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    powsimp_index_list = judge_powsimp_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(IR, powsimp_index_list,has_transformation, original_IR_list, transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    _,one_step_transformed_IR_list,_=apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    # print(f'one_step_transformed_IR_list:{one_step_transformed_IR_list}')
    for one_step_transformed_IR_idx in range(len(one_step_transformed_IR_list)):
        one_step_transformed_IR=one_step_transformed_IR_list[one_step_transformed_IR_idx]
        # print(f'one_step_transformed_IR: {one_step_transformed_IR}')
        one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops,one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops = split_IR_to_equations(one_step_transformed_IR)
        # print(f'one_step_row_equations_under_loops:{one_step_row_equations_under_loops},one_step_equations_under_loops:{one_step_equations_under_loops}\none_step_eq_inputs_under_loops:{one_step_eq_inputs_under_loops}')
        _,two_step_transformed_IR_list,_=apply_compute_inline_to_IR(one_step_transformed_IR, input_output_name, name_start_idx, one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops, one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        # print(f'two_step_transformed_IR_list:{two_step_transformed_IR_list}')
        for two_step_transformed_IR_idx in range(len(two_step_transformed_IR_list)):
            two_step_transformed_IR=two_step_transformed_IR_list[two_step_transformed_IR_idx]
            # print(f'two_step_transformed_IR: {two_step_transformed_IR}')
            two_step_row_equations_under_loops, two_step_loops, two_step_equations_under_loops, two_step_eq_outputs_under_loops,two_step_eq_inputs_under_loops, two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops = split_IR_to_equations(two_step_transformed_IR)
            two_step_powsimp_index_list = judge_powsimp_condition(two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops)
            has_transformation,original_IR_list,  transformed_IR_list=apply_sympy_to_IR(two_step_transformed_IR, two_step_powsimp_index_list,has_transformation,original_IR_list, transformed_IR_list, two_step_loops, two_step_row_equations_under_loops, two_step_equations_under_loops, two_step_simplified_eq_inputs_under_loops, two_step_eq_inputs_under_loops)
        one_step_powsimp_index_list = judge_powsimp_condition(one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(one_step_transformed_IR, one_step_powsimp_index_list, has_transformation, original_IR_list, transformed_IR_list, one_step_loops, one_step_row_equations_under_loops, one_step_equations_under_loops, one_step_simplified_eq_inputs_under_loops, one_step_eq_inputs_under_loops)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_expand_powsimp_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    original_IR, transformed_IR_list, has_transformation=apply_powsimp_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    return transformed_IR_list, original_IR, has_transformation

def apply_expand_log_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    expand_log_index_list = judge_expand_log_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(IR, expand_log_index_list,has_transformation,original_IR_list, transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    _,one_step_transformed_IR_list,_=apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    # print(f'one_step_transformed_IR_list:{one_step_transformed_IR_list}')
    for one_step_transformed_IR_idx in range(len(one_step_transformed_IR_list)):
        one_step_transformed_IR=one_step_transformed_IR_list[one_step_transformed_IR_idx]
        # print(f'one_step_transformed_IR: {one_step_transformed_IR}')
        one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops,one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops = split_IR_to_equations(one_step_transformed_IR)
        # print(f'one_step_row_equations_under_loops:{one_step_row_equations_under_loops},one_step_equations_under_loops:{one_step_equations_under_loops}\none_step_eq_inputs_under_loops:{one_step_eq_inputs_under_loops}')
        _,two_step_transformed_IR_list,_=apply_compute_inline_to_IR(one_step_transformed_IR, input_output_name, name_start_idx, one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops, one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        # print(f'two_step_transformed_IR_list:{two_step_transformed_IR_list}')
        for two_step_transformed_IR_idx in range(len(two_step_transformed_IR_list)):
            two_step_transformed_IR=two_step_transformed_IR_list[two_step_transformed_IR_idx]
            # print(f'two_step_transformed_IR: {two_step_transformed_IR}')
            two_step_row_equations_under_loops, two_step_loops, two_step_equations_under_loops, two_step_eq_outputs_under_loops,two_step_eq_inputs_under_loops, two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops = split_IR_to_equations(two_step_transformed_IR)
            two_step_expand_log_index_list = judge_expand_log_condition(two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops)
            has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(two_step_transformed_IR, two_step_expand_log_index_list,has_transformation,original_IR_list, transformed_IR_list, two_step_loops, two_step_row_equations_under_loops, two_step_equations_under_loops, two_step_simplified_eq_inputs_under_loops, two_step_eq_inputs_under_loops)
        one_step_expand_log_index_list = judge_expand_log_condition(one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        has_transformation, original_IR_list, transformed_IR_list=apply_sympy_to_IR(one_step_transformed_IR, one_step_expand_log_index_list, has_transformation, original_IR_list, transformed_IR_list, one_step_loops, one_step_row_equations_under_loops, one_step_equations_under_loops, one_step_simplified_eq_inputs_under_loops, one_step_eq_inputs_under_loops)
    return original_IR_list,  transformed_IR_list, has_transformation

def apply_logsimp_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    original_IR, transformed_IR_list, has_transformation=apply_expand_log_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    return transformed_IR_list, original_IR, has_transformation

def apply_collect_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    collect_index_list = judge_collect_condition(simplified_eqs_under_loops,simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    origianl_IR_list=[]
    has_transformation, origianl_IR_list, transformed_IR_list=apply_sympy_to_IR(IR, collect_index_list,has_transformation, origianl_IR_list, transformed_IR_list, loops, row_equations_under_loops, equations_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    _,one_step_transformed_IR_list,_=apply_compute_inline_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    # print(f'one_step_transformed_IR_list:{one_step_transformed_IR_list}')
    for one_step_transformed_IR_idx in range(len(one_step_transformed_IR_list)):
        one_step_transformed_IR=one_step_transformed_IR_list[one_step_transformed_IR_idx]
        # print(f'one_step_transformed_IR: {one_step_transformed_IR}')
        one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops,one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops = split_IR_to_equations(one_step_transformed_IR)
        # print(f'one_step_row_equations_under_loops:{one_step_row_equations_under_loops},one_step_equations_under_loops:{one_step_equations_under_loops}\none_step_eq_inputs_under_loops:{one_step_eq_inputs_under_loops}')
        _,two_step_transformed_IR_list,_=apply_compute_inline_to_IR(one_step_transformed_IR, input_output_name, name_start_idx, one_step_row_equations_under_loops, one_step_loops, one_step_equations_under_loops, one_step_eq_outputs_under_loops, one_step_eq_inputs_under_loops, one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        # print(f'two_step_transformed_IR_list:{two_step_transformed_IR_list}')
        for two_step_transformed_IR_idx in range(len(two_step_transformed_IR_list)):
            two_step_transformed_IR=two_step_transformed_IR_list[two_step_transformed_IR_idx]
            # print(f'two_step_transformed_IR: {two_step_transformed_IR}')
            two_step_row_equations_under_loops, two_step_loops, two_step_equations_under_loops, two_step_eq_outputs_under_loops,two_step_eq_inputs_under_loops, two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops = split_IR_to_equations(two_step_transformed_IR)
            two_step_collect_index_list = judge_collect_condition(two_step_simplified_eqs_under_loops, two_step_simplified_eq_outputs_under_loops, two_step_simplified_eq_inputs_under_loops)
            has_transformation, origianl_IR_list, transformed_IR_list=apply_sympy_to_IR(two_step_transformed_IR, two_step_collect_index_list,has_transformation,origianl_IR_list, transformed_IR_list, two_step_loops, two_step_row_equations_under_loops, two_step_equations_under_loops, two_step_simplified_eq_inputs_under_loops, two_step_eq_inputs_under_loops)
        one_step_collect_index_list = judge_collect_condition(one_step_simplified_eqs_under_loops, one_step_simplified_eq_outputs_under_loops, one_step_simplified_eq_inputs_under_loops)
        has_transformation, origianl_IR_list, transformed_IR_list=apply_sympy_to_IR(one_step_transformed_IR, one_step_collect_index_list, has_transformation, origianl_IR_list, transformed_IR_list, one_step_loops, one_step_row_equations_under_loops, one_step_equations_under_loops, one_step_simplified_eq_inputs_under_loops, one_step_eq_inputs_under_loops)
    return origianl_IR_list,  transformed_IR_list, has_transformation

def apply_expand_collect_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    original_IR, transformed_IR_list, has_transformation=apply_collect_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    return transformed_IR_list, original_IR, has_transformation

def apply_partially_equivalent_then_correct_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    enable_partially_equivalent_then_correct=judge_partially_equivalent_then_correct_condition(loops,simplified_eqs_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    # print(f'enable_partially_equivalent_then_correct: {enable_partially_equivalent_then_correct}')
    if enable_partially_equivalent_then_correct:
        #concat input
        concat_values_list, concat_keys_list, concat_loop_types_list,_=split_loops_into_value_and_index([loops[0]])
        loop1_values_list, _, _,_=split_loops_into_value_and_index([loops[1]])
        concat_var_name, name_start_idx = generate_names(1,name_start_idx)
        concat_input_superscript=re.findall(r'\^{[a-zA-Z0-9,]*}', eq_inputs_under_loops[0][0])[0]
        first_concat_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(eq_inputs_under_loops[0][0])
        first_concat_subscript_details=generate_subscript_details(first_concat_input_subscript_list)
        padding_size=[int(''.join(item.split('-')[1:])) for item in first_concat_subscript_details if '-' in item]
        kernel_size=loop1_values_list[0][-(len(concat_values_list[0])-2):]
        # print(f'concat_values_list:{concat_values_list}')
        for concat_axis in list(range(2,len(concat_values_list[0]))):
            new_first_concat_subscript_details=[]
            second_concat_subscript_details=[]
            new_concat_values_list=concat_values_list[0][:2]
            for value_index in range(2,len(concat_values_list[0])):
                new_concat_values_list.append(concat_values_list[0][value_index]-padding_size[value_index-2]*2)
            for first_concat_key in first_concat_subscript_details:
                if concat_keys_list[0][concat_axis] not in first_concat_key:
                    second_concat_subscript_details.append(first_concat_key.split('-')[0])
                    new_first_concat_subscript_details.append(first_concat_key.split('-')[0])
                else:
                    second_concat_subscript_details.append(first_concat_key.split('-')[0]+'+'+str(new_concat_values_list[concat_axis]))
                    new_first_concat_subscript_details.append(first_concat_key.split('-')[0])
            first_concat_input_subscript='_{'+','.join(new_first_concat_subscript_details)+'}'
            second_concat_input_subscript='_{'+','.join(second_concat_subscript_details)+'}'
            concat_loop_notation=curate_loops(new_concat_values_list,concat_keys_list[0],concat_loop_types_list[0])
            concat_input_eq=concat_loop_notation+'['+concat_var_name[0]+concat_input_superscript+first_concat_input_subscript+'='+simplified_eq_inputs_under_loops[0][0]+concat_input_superscript+first_concat_input_subscript+';'+concat_var_name[0]+concat_input_superscript+second_concat_input_subscript+'='+simplified_eq_inputs_under_loops[2][0]+concat_input_superscript+first_concat_input_subscript+';];'
            # print(f'concat_axis:{concat_axis}, concat_input_eq:{concat_input_eq}')
            #conv
            second_conv_values_list, second_conv_keys_list, second_conv_loop_types_list,_=split_loops_into_value_and_index([loops[1]])
            first_conv_loop_values_list=concat_values_list[0][:concat_axis]+[(concat_values_list[0][concat_axis]-padding_size[concat_axis-2])*2]+concat_values_list[0][concat_axis+1:]
            first_conv_loop_notation=curate_loops(first_conv_loop_values_list,concat_keys_list[0],concat_loop_types_list[0])
            conv_var_name, name_start_idx = generate_names(2,name_start_idx)
            first_conv_output_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[0][0])
            second_conv_output_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[1][0])
            kernel_input_list=[item for item in eq_inputs_under_loops[1] if item not in eq_outputs_under_loops[1] and simplified_eq_outputs_under_loops[0][0] not in item]
            second_conv_input_list=[item for item in eq_inputs_under_loops[1] if simplified_eq_outputs_under_loops[0][0] in item]
            second_conv_input_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(second_conv_input_list[0])
            second_conv_input_subscript_details=generate_subscript_details(second_conv_input_subscript_list)
            stride_dilation_subscript=second_conv_input_subscript_details[concat_axis].split('+')
            if '*' not in second_conv_input_subscript_details[concat_axis-1]:
                stride,dilation=1,1
            else:
                if second_conv_keys_list[0][concat_axis] in stride_dilation_subscript[0]:
                    stride=int(''.join(stride_dilation_subscript[0].split('*')[1:]))
                    dilation=int(''.join(stride_dilation_subscript[1].split('*')[1:]))
                else:
                    stride=int(''.join(stride_dilation_subscript[1].split('*')[1:]))
                    dilation=int(''.join(stride_dilation_subscript[0].split('*')[1:]))
            # print(f'subscript_details:{second_conv_input_subscript_list}, subscript:{second_conv_input_subscript_details[concat_axis-1]}, stride:{stride}, dilation:{dilation}')
            second_concat_value_without_stride=((concat_values_list[0][concat_axis]-padding_size[concat_axis-2])*2-(second_conv_values_list[0][concat_axis-len(concat_values_list[0])]-1)*dilation)
            if second_concat_value_without_stride%stride==0:
                has_transformation=True
                second_concat_value=second_concat_value_without_stride/stride
                second_conv_loop_values_list=second_conv_values_list[0][:concat_axis]+[int(second_concat_value)]+second_conv_values_list[0][concat_axis+1:]
                second_conv_loop_notation=curate_loops(second_conv_loop_values_list,second_conv_keys_list[0],second_conv_loop_types_list[0])
                # print(f'first_simplified_input:{simplified_eq_outputs_under_loops[0][0]}, eq_inputs_under_loops[1]:{eq_inputs_under_loops[1]}\nsecond_conv_input_list:{second_conv_input_list}')
                second_conv_input_subscript_list,_=find_subscripts_of_input_output_and_simplified_version(second_conv_input_list[0])
                first_conv_eq=first_conv_loop_notation+'['+conv_var_name[0]+concat_input_superscript+first_conv_output_subscript_list[0]+'=if_then_else('
                for idx in range(len(padding_size)):
                    if idx<len(padding_size)-1:
                        first_conv_eq+=str(padding_size[idx])+'<='+concat_keys_list[0][idx+2]+'<'+str(first_conv_loop_values_list[idx+2]-padding_size[idx])+'&'
                    else:
                        first_conv_eq+=str(padding_size[idx])+'<='+concat_keys_list[0][idx+2]+'<'+str(first_conv_loop_values_list[idx+2]-padding_size[idx])+','
                first_conv_eq+=concat_var_name[0]+concat_input_superscript+first_concat_input_subscript_list[0]+','+equations_under_loops[0][0].split(',')[-1]+';];'
                second_conv_eq=second_conv_loop_notation+'['+conv_var_name[1]+concat_input_superscript+second_conv_output_subscript_list[0]+'='+conv_var_name[1]+concat_input_superscript+second_conv_output_subscript_list[0]+'+'+conv_var_name[0]+concat_input_superscript+second_conv_input_subscript_list[0]+'*'+kernel_input_list[0]+';];'
                conv_eq=first_conv_eq+second_conv_eq
                # print(f'concat_axis:{concat_axis}, conv_eq:{conv_eq}')
                #split output
                # print(f'second_conv_loop_values_list:{second_conv_loop_values_list}, index:{4-len(concat_values_list[0])}')
                # print(f'new_first_concat_subscript_details:{new_first_concat_subscript_details}')
                minus_value=loop1_values_list[0][concat_axis]-second_conv_loop_values_list[concat_axis]/2
                # print(f'loop1_values_list[0][concat_axis]:{loop1_values_list[0][concat_axis]},second_conv_loop_values_list[concat_axis]/2:{second_conv_loop_values_list[concat_axis]/2}')
                if minus_value>=0:
                    split_values_list=second_conv_loop_values_list[:concat_axis]+[second_conv_loop_values_list[concat_axis]//2]+loop1_values_list[0][concat_axis+1:len(concat_values_list[0])]
                    if minus_value==0:
                        new_first_split_subscript_details=new_first_concat_subscript_details
                    else:
                        new_first_split_subscript_details=new_first_concat_subscript_details[:concat_axis]+[new_first_concat_subscript_details[concat_axis]+'+'+str(math.ceil((minus_value)/2))]+new_first_concat_subscript_details[concat_axis+1:]
                    split_loop_notation=curate_loops(split_values_list,concat_keys_list[0],concat_loop_types_list[0])
                    new_second_split_input_subscript_details=new_first_concat_subscript_details[:concat_axis]+[new_first_concat_subscript_details[concat_axis]+'+'+str(int(second_conv_loop_values_list[concat_axis]/2))]+new_first_concat_subscript_details[concat_axis+1:]
                    new_first_split_subscript='_{'+','.join(new_first_split_subscript_details)+'}'
                    new_second_split_input_subscript='_{'+','.join(new_second_split_input_subscript_details)+'}'
                    temp_second_split_output=eq_outputs_under_loops[3][0]
                    temp_second_split_output=temp_second_split_output[:temp_second_split_output.index('_')]
                    second_split_output=temp_second_split_output+new_first_split_subscript
                    split_eq=split_loop_notation+'['+eq_outputs_under_loops[1][0]+'='+conv_var_name[1]+concat_input_superscript+second_conv_output_subscript_list[0]+';'+second_split_output+'='+conv_var_name[1]+concat_input_superscript+new_second_split_input_subscript+';];'
                else:
                    new_first_split_subscript_details=new_first_concat_subscript_details[:concat_axis]+[new_first_concat_subscript_details[concat_axis]]+new_first_concat_subscript_details[concat_axis+1:]
                    split_values_list1=second_conv_loop_values_list[:concat_axis]+[loop1_values_list[0][concat_axis]]+loop1_values_list[0][concat_axis+1:len(concat_values_list[0])]
                    # print(f'minus_value:{minus_value},int(second_conv_loop_values_list[concat_axis]/2):{int(second_conv_loop_values_list[concat_axis]/2)}')
                    split_loop_notation1=curate_loops(split_values_list1,concat_keys_list[0],concat_loop_types_list[0])
                    new_second_split_input_subscript_details=new_first_concat_subscript_details[:concat_axis]+[new_first_concat_subscript_details[concat_axis]+'+'+str(int(second_conv_loop_values_list[concat_axis]/2)+int(-minus_value))]+new_first_concat_subscript_details[concat_axis+1:]
                    new_first_split_subscript='_{'+','.join(new_first_split_subscript_details)+'}'
                    new_second_split_input_subscript='_{'+','.join(new_second_split_input_subscript_details)+'}'
                    temp_second_split_output=eq_outputs_under_loops[3][0]
                    temp_second_split_output=temp_second_split_output[:temp_second_split_output.index('_')]
                    second_split_output=temp_second_split_output+new_first_split_subscript
                    split_eq=split_loop_notation1+'['+eq_outputs_under_loops[1][0]+'='+conv_var_name[1]+concat_input_superscript+second_conv_output_subscript_list[0]+';'+second_split_output+'='+conv_var_name[1]+concat_input_superscript+new_second_split_input_subscript+';];'
                # print(f'concat_axis:{concat_axis}, split_eq:{split_eq}')
                #correct
                # print(f'second_conv_values_list[0]:{second_conv_values_list[0]},kernel_size:{kernel_size}')
                correct_eq=row_equations_under_loops[0]+row_equations_under_loops[2]
                correct_values_list=second_conv_values_list[0][:concat_axis]+[kernel_size[concat_axis-2]-1]+second_conv_values_list[0][concat_axis+1:]
                correct_loop_notation_1=curate_loops(correct_values_list[:len(concat_values_list[0])],second_conv_keys_list[0][:len(concat_values_list[0])],second_conv_loop_types_list[0][:len(concat_values_list[0])])
                correct_loop_notation_2=curate_loops(correct_values_list[len(concat_values_list[0]):],second_conv_keys_list[0][len(concat_values_list[0]):],second_conv_loop_types_list[0][len(concat_values_list[0]):])
                correct_output_subscript_details=generate_subscript_details(second_conv_output_subscript_list)
                if minus_value>=0:
                    first_correct_stride_part=second_conv_keys_list[0][concat_axis]+'+'+str(split_values_list[concat_axis]-kernel_size[concat_axis-2]+2)
                else:
                    first_correct_stride_part=second_conv_keys_list[0][concat_axis]+'+'+str(split_values_list1[concat_axis]-kernel_size[concat_axis-2]+1)
                # print(f'first_correct_stride_part:{first_correct_stride_part}')
                second_correct_stride_part=str(kernel_size[concat_axis-2]-2)+'-'+second_conv_keys_list[0][concat_axis]
                first_correct_output_subscript_details=correct_output_subscript_details[:concat_axis]+[first_correct_stride_part]+correct_output_subscript_details[concat_axis+1:]
                second_correct_output_subscript_details=correct_output_subscript_details[:concat_axis]+[second_correct_stride_part]+correct_output_subscript_details[concat_axis+1:]
                first_correct_output_subscript='_{'+','.join(first_correct_output_subscript_details)+'}'
                second_correct_output_subscript='_{'+','.join(second_correct_output_subscript_details)+'}'
                # print(f'first_correct_stride_part:{first_correct_stride_part}, second_correct_stride_part:{second_correct_stride_part},stride_dilation_subscript:{stride_dilation_subscript}')
                if second_conv_keys_list[0][concat_axis] in stride_dilation_subscript[0]:
                    if stride>1:
                        first_correct_concat_subscript='('+first_correct_stride_part+')*'+str(stride)+'+'+stride_dilation_subscript[1]
                        second_correct_concat_subscript='('+second_correct_stride_part+')*'+str(stride)+'+'+stride_dilation_subscript[1]
                    else:
                        first_correct_concat_subscript='('+first_correct_stride_part+')'+'+'+stride_dilation_subscript[1]
                        second_correct_concat_subscript='('+second_correct_stride_part+')'+'+'+stride_dilation_subscript[1]
                else:
                    if stride>1:
                        first_correct_concat_subscript=stride_dilation_subscript[0]+'+'+'('+first_correct_stride_part+')*'+str(stride)
                        second_correct_concat_subscript=stride_dilation_subscript[0]+'+'+'('+second_correct_stride_part+')*'+str(stride)
                    else:
                        first_correct_concat_subscript=stride_dilation_subscript[0]+'+'+'('+first_correct_stride_part+')'
                        second_correct_concat_subscript=stride_dilation_subscript[0]+'+'+'('+second_correct_stride_part+')'
                first_correct_input_subscript_details=second_conv_input_subscript_details[:concat_axis]+[first_correct_concat_subscript]+second_conv_input_subscript_details[concat_axis+1:]
                second_correct_input_subscript_details=second_conv_input_subscript_details[:concat_axis]+[second_correct_concat_subscript]+second_conv_input_subscript_details[concat_axis+1:]
                first_correct_input_subscript='_{'+','.join(first_correct_input_subscript_details)+'}'
                second_correct_input_subscript='_{'+','.join(second_correct_input_subscript_details)+'}'
                correct_eq+=correct_loop_notation_1+'['+simplified_eq_outputs_under_loops[1][0]+concat_input_superscript+first_correct_output_subscript+'=0;]'
                correct_eq+=correct_loop_notation_2+'['+simplified_eq_outputs_under_loops[1][0]+concat_input_superscript+first_correct_output_subscript+'='+simplified_eq_outputs_under_loops[1][0]+concat_input_superscript+first_correct_output_subscript+'+'+simplified_eq_outputs_under_loops[0][0]+concat_input_superscript+first_correct_input_subscript+'*'+kernel_input_list[0]+';];'
                correct_eq+=correct_loop_notation_1+'['+simplified_eq_outputs_under_loops[3][0]+concat_input_superscript+second_correct_output_subscript+'=0;]'
                correct_eq+=correct_loop_notation_2+'['+simplified_eq_outputs_under_loops[3][0]+concat_input_superscript+second_correct_output_subscript+'='+simplified_eq_outputs_under_loops[3][0]+concat_input_superscript+second_correct_output_subscript+'+'+simplified_eq_outputs_under_loops[2][0]+concat_input_superscript+second_correct_input_subscript+'*'+kernel_input_list[0]+';];'
                # print(f'concat_axis:{concat_axis}, correct_eq:{correct_eq}')
                #update IR
                transformed_IR_list.append(concat_input_eq+conv_eq+split_eq+correct_eq+''.join(row_equations_under_loops[4:]))
                original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_normal_loop_max_to_prefix_max_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    normal_loop_max_to_prefix_max_index_list=judge_normal_loop_max_to_prefix_max_condition(simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    for op_index in normal_loop_max_to_prefix_max_index_list:
        # print(f'op_index:{op_index},row_eq:{row_equations_under_loops[op_index]}')
        this_loop=loops[op_index]
        intermediate_vars, name_start_idx=generate_names(1, name_start_idx)
        values_list, keys_list, loop_types_list,_=split_loops_into_value_and_index([this_loop])
        this_output=simplified_eq_outputs_under_loops[op_index][0]
        this_input_list=[item for item in eq_inputs_under_loops[op_index] if this_output not in item]
        this_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(this_input_list[0])
        this_input_subscript_details=generate_subscript_details(this_input_subscript_list)
        this_input_superscript=re.findall(r'\^{[a-zA-Z0-9,]*}', this_input_list[0])[0]
        this_output_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(eq_outputs_under_loops[op_index][0])
        this_output_subscript_details=[]
        second_input_subscript_details=[]
        new_keys_index=[]
        has_tx=False
        related_idx=''
        for item in this_input_subscript_details:
            if item in this_output_subscript_list[0]:
                this_output_subscript_details.append(item)
                second_input_subscript_details.append(item)
                if 'tx' in item:
                    new_keys_index.append(keys_list[0].index('tx'))
                    has_tx=True
                else:
                    new_keys_index.append(keys_list[0].index(item))
            else:
                related_idx=item+'-1'
                this_output_subscript_details.append(related_idx)
                if 'tx' in item:
                    temp_index=keys_list[0].index('tx')
                else:
                    temp_index=keys_list[0].index(item)
                second_input_subscript_details.append(str(values_list[0][temp_index]-1))
        this_output_subscript='_{'+','.join(this_output_subscript_details)+'}'
        second_input_subscript='_{'+','.join(second_input_subscript_details)+'}'
        if has_tx:
            new_values_list=[values_list[0][idx] for idx in new_keys_index]
            new_keys_list=[keys_list[0][idx] for idx in new_keys_index]
            new_loop_types_list=[loop_types_list[0][idx] for idx in new_keys_index]
        else:
            new_values_list=[values_list[0][idx] for idx in new_keys_index]
            new_keys_list=['tx']+[keys_list[0][idx] for idx in new_keys_index[1:]]
            new_loop_types_list=['B']+[loop_types_list[0][idx] for idx in new_keys_index[1:]]
        new_loop_notation=curate_loops(new_values_list, new_keys_list, new_loop_types_list)
        intermediate_part='if_then_else('+related_idx+'<0,-inf,'+intermediate_vars[0]+this_input_superscript+this_output_subscript+')'
        new_eq=this_loop+'['+intermediate_vars[0]+this_input_superscript+this_input_subscript_list[0]+'=max('+intermediate_part+','+this_input_list[0]+');];'
        # print(f'new_eq:{new_eq}')
        new_eq+=new_loop_notation+'['+eq_outputs_under_loops[op_index][0]+'='+intermediate_vars[0]+this_input_superscript+second_input_subscript+';];'
        transformed_IR=''.join(row_equations_under_loops[:op_index]) + new_eq + ''.join(row_equations_under_loops[op_index+1:])
        transformed_IR_list.append(transformed_IR)
        original_IR_list.append(IR)
        has_transformation = True
        # print(f'row_eq:{row_equations_under_loops[op_index]}\nnew_eq:{new_eq}')
    return original_IR_list, transformed_IR_list, has_transformation

def apply_exponential_split_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    exponential_split_index_list=judge_exponential_split_condition(loops, simplified_eqs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    for index in range(len(exponential_split_index_list)):
        op_index, eq_index = exponential_split_index_list[index]
        has_transformation = True
        this_simplified_eq=simplified_eqs_under_loops[op_index][eq_index]
        exp_parts=re.findall(r'exp\(.*?\)',this_simplified_eq)
        # print(f'exp_parts:{exp_parts}')
        for exp_part in exp_parts:
            simplified_split_exp_part=re.findall(r'[A-Za-z]+|[^a-zA-Z]',exp_part.replace('exp',''))
            split_exp_part=[]
            full_inputs_in_exp_part=[]
            for item in simplified_split_exp_part:
                if item in simplified_eq_inputs_under_loops[op_index]:
                    full_inputs=[full_item for full_item in eq_inputs_under_loops[op_index] if re.sub(r'\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', full_item)==item or re.sub(r'\^{.*?}','', full_item)==item]
                    full_inputs_in_exp_part.append(full_inputs[0])
                    split_exp_part.append(full_inputs[0])
                else:
                    split_exp_part.append(item)
            full_exp_part=''.join(['exp']+split_exp_part)
            other_part=equations_under_loops[op_index][eq_index].split(full_exp_part)
            first_input_superscript=re.findall(r'\^{[a-zA-Z0-9,]*}', full_inputs_in_exp_part[0])[0]
            first_input_subscript_list, first_simplified_input=find_subscripts_of_input_output_and_simplified_version(full_inputs_in_exp_part[0])
            first_input_subscript_details=generate_subscript_details(first_input_subscript_list)
            new_subscript_details=first_input_subscript_details[:-1]+[first_input_subscript_details[-1]+'-1']
            new_subscript='_{'+','.join(new_subscript_details)+'}'
            temp_new_input_list=['if_then_else('+first_input_subscript_details[-1]+'-1<0,0,'+first_simplified_input+first_input_superscript+new_subscript+')']
            this_loop=loops[op_index]
            this_output=eq_outputs_under_loops[op_index][0]
            this_output_subscript=this_output[this_output.index('_'):] if '_' in this_output else ''
            this_output_subscript_details=re.findall(rf'[a-z]+', this_output_subscript)
            for idx in range(op_index):
                if loops[idx]==this_loop:
                    for iitem in eq_outputs_under_loops[idx]:
                        if iitem!=this_output and 'exp' not in row_equations_under_loops[idx] and 'inf' not in row_equations_under_loops[idx] and '_{0}' not in row_equations_under_loops[idx] and 'if_then_else' not in row_equations_under_loops[idx] and 'erf' not in row_equations_under_loops[idx]:
                            temp_new_input_list.append(iitem)
                if 'exp' in row_equations_under_loops[idx] or 'inf' in row_equations_under_loops[idx] or '_{0}' in row_equations_under_loops[idx] or 'if_then_else' in row_equations_under_loops[idx] or 'erf' in row_equations_under_loops[idx]:
                    simplified_output=simplified_eq_outputs_under_loops[idx]
                    remove_elem=[iitem for item in simplified_output for iitem in temp_new_input_list if item+'^' in iitem]
                    for elem in remove_elem:
                        temp_new_input_list.remove(elem)
            new_input_list=[]
            for new_input in temp_new_input_list:
                temp_new_input=new_input.replace('if_then_else','')
                new_input_subscript=temp_new_input[temp_new_input.index('_'):] if '_' in temp_new_input else ''
                new_input_subscript_details=re.findall(rf'[a-z]+', new_input_subscript)
                # print(f'new_input:{new_input},new_input_subscript_details:{new_input_subscript_details},this_output_subscript_details:{this_output_subscript_details}')
                if len(this_output_subscript_details)==0 and 'tx' in temp_new_input:
                    continue
                if len(new_input_subscript_details)==len(set(this_output_subscript_details+new_input_subscript_details))  and '^{bool,' not in temp_new_input:
                    new_input_list.append(new_input)
            for new_input_idx in range(len(new_input_list)):
                old_input_index=split_exp_part.index(full_inputs_in_exp_part[0])
                new_input=new_input_list[new_input_idx]
                new_eq1='exp'+''.join(split_exp_part[:old_input_index+1])+'-'+new_input+')'
                if new_input_idx==0:
                    use_in_multiply = bool(random.getrandbits(1))
                if (new_input_idx==0 and use_in_multiply) or new_input_idx>0:
                    new_eq2='exp'+'('+new_input+''.join(split_exp_part[old_input_index+1:])
                    new_eq_multiply=new_eq1+'*'+new_eq2
                    final_new_eq_multiply=new_eq_multiply.join(other_part)
                    if eq_index>0:
                        transformed_eqs=this_loop+'['+';'.join(equations_under_loops[op_index][:eq_index])+';'+final_new_eq_multiply+';'.join(equations_under_loops[op_index][eq_index+1:])+';];'
                    else:
                        transformed_eqs=this_loop+'['+final_new_eq_multiply+';'.join(equations_under_loops[op_index][eq_index+1:])+';];'
                    print(f'multiply_transformed_eqs:{transformed_eqs}')
                    transformed_IR=''.join(row_equations_under_loops[:op_index]) + transformed_eqs + ''.join(row_equations_under_loops[op_index+1:])
                    transformed_IR_list.append(transformed_IR)
                    original_IR_list.append(IR)
                if (new_input_idx==0 and not use_in_multiply) or new_input_idx>0:
                    if split_exp_part[old_input_index+1]=='-':
                        if len(split_exp_part[old_input_index+2:-1])==1:
                            new_eq3='exp('+split_exp_part[old_input_index+2]+'-'+new_input+')'
                        else:
                            new_eq3='exp(-('+''.join(split_exp_part[old_input_index+2:-1])+')-'+new_input+')'
                        new_eq_division=new_eq1+'/'+new_eq3
                        final_new_eq_division=new_eq_division.join(other_part)
                        if eq_index>0:
                            transformed_eqs=this_loop+'['+';'.join(equations_under_loops[op_index][:eq_index])+';'+final_new_eq_division+';'.join(equations_under_loops[op_index][eq_index+1:])+';];'
                        else:
                            transformed_eqs=this_loop+'['+final_new_eq_division+';'.join(equations_under_loops[op_index][eq_index+1:])+';];'
                        print(f'division_transformed_eqs:{transformed_eqs}')
                        transformed_IR=''.join(row_equations_under_loops[:op_index]) + transformed_eqs + ''.join(row_equations_under_loops[op_index+1:])
                        transformed_IR_list.append(transformed_IR)
                        original_IR_list.append(IR)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_multiplicative_split_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops, split_sig='multiply'):
    multiplicative_split_index_list=judge_multiplicative_split_condition(loops, simplified_eqs_under_loops, simplified_eq_inputs_under_loops, eq_inputs_under_loops)
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    if_then_else_transform_idx=[]
    for index in range(len(multiplicative_split_index_list)):
        op_index, eq_index, simplified_input = multiplicative_split_index_list[index]
        # print(f'op_index:{op_index}, eq_index:{eq_index}, input:{simplified_input}, eq:{simplified_eqs_under_loops[op_index][eq_index]}, full_eq:{equations_under_loops[op_index][eq_index]}')
        this_simplified_eq=simplified_eqs_under_loops[op_index][eq_index]
        simplified_split_eq=re.findall(r'[A-Za-z]+|[^a-zA-Z]',this_simplified_eq)
        split_eq=[]
        full_input_in_eq=0
        for item in simplified_split_eq:
            if item in simplified_eq_inputs_under_loops[op_index]:
                full_inputs=[full_item for full_item in eq_inputs_under_loops[op_index] if re.sub(r'\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', full_item)==item or re.sub(r'\^{.*?}','', full_item)==item]
                split_eq.append(full_inputs[0])
                if item==simplified_input:
                    full_input_in_eq=full_inputs[0]
            else:
                if item in simplified_eq_outputs_under_loops[op_index]:
                   full_outputs=[full_item for full_item in eq_outputs_under_loops[op_index] if re.sub(r'\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', full_item)==item or re.sub(r'\^{.*?}','', full_item)==item]
                   split_eq.append(full_outputs[0]) 
                else:
                    split_eq.append(item)
        # print(f'split_eq:{split_eq}')
        first_input_superscript=re.findall(r'\^{[a-zA-Z0-9,]*}', full_input_in_eq)[0]
        first_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(full_input_in_eq)
        first_input_subscript_details=generate_subscript_details(first_input_subscript_list)
        new_subscript_details=first_input_subscript_details[:-1]+[first_input_subscript_details[-1]+'-1']
        new_subscript='_{'+','.join(new_subscript_details)+'}'
        old_input_index=split_eq.index(full_input_in_eq)
        if split_sig=='add':
            temp_new_input_list=['if_then_else('+first_input_subscript_details[-1]+'-1<0,0,'+simplified_input+first_input_superscript+new_subscript+')']
        else:
            temp_new_input_list=['if_then_else('+first_input_subscript_details[-1]+'-1<0,1,'+simplified_input+first_input_superscript+new_subscript+')']
        this_loop=loops[op_index]
        this_output=eq_outputs_under_loops[op_index][0]
        this_output_subscript=this_output[this_output.index('_'):] if '_' in this_output else ''
        this_output_subscript_details=re.findall(rf'[a-z]+', this_output_subscript)
        for idx in range(op_index):
            if loops[idx]==this_loop:
                for iitem in eq_outputs_under_loops[idx]:
                    if iitem!=this_output and 'exp' not in row_equations_under_loops[idx] and 'inf' not in row_equations_under_loops[idx] and '_{0}' not in row_equations_under_loops[idx] and 'if_then_else' not in row_equations_under_loops[idx] and 'erf' not in row_equations_under_loops[idx]:
                        if split_sig=='multiply' and ',0)' not in row_equations_under_loops[idx] and '-' not in ';'.join(simplified_eqs_under_loops[idx]):
                            temp_new_input_list.append(iitem)
                        else:
                            temp_new_input_list.append(iitem)
            if 'exp' in row_equations_under_loops[idx] or 'inf' in row_equations_under_loops[idx] or '_{0}' in row_equations_under_loops[idx] or (split_sig=='multiply' and (',0)' in row_equations_under_loops[idx] or '-' in ';'.join(simplified_eqs_under_loops[idx]))) or 'if_then_else' in row_equations_under_loops[idx] or 'erf' in row_equations_under_loops[idx]:
                simplified_output=simplified_eq_outputs_under_loops[idx]
                remove_elem=[iitem for item in simplified_output for iitem in temp_new_input_list if item+'^' in iitem]
                for elem in remove_elem:
                    temp_new_input_list.remove(elem)
        new_input_list=[]
        for new_input in temp_new_input_list:
            temp_new_input=new_input.replace('if_then_else','')
            new_input_subscript=temp_new_input[temp_new_input.index('_'):] if '_' in temp_new_input else ''
            new_input_subscript_details=re.findall(rf'[a-z]+', new_input_subscript)
            # print(f'new_input:{new_input},new_input_subscript_details:{new_input_subscript_details},this_output_subscript_details:{this_output_subscript_details}')
            if len(this_output_subscript_details)==0 and 'tx' in temp_new_input:
                continue
            if len(new_input_subscript_details)==len(set(this_output_subscript_details+new_input_subscript_details)) and '^{bool,' not in temp_new_input:
                new_input_list.append(new_input)
        for new_input in new_input_list:
            if split_sig=='multiply':
                new_part=full_input_in_eq+'/'+new_input+'*'+new_input
            elif split_sig=='add':
                new_part=full_input_in_eq+'+'+new_input+'-'+new_input
            if split_eq[old_input_index-1]=='-' or split_eq[old_input_index-1]=='*' or split_eq[old_input_index-1]=='/' or (old_input_index<len(split_eq)-1 and split_eq[old_input_index+1]=='*') or (old_input_index<len(split_eq)-1 and split_eq[old_input_index+1]=='/'):
                new_eq=''.join(split_eq[:old_input_index])+'('+new_part+')'+''.join(split_eq[old_input_index+1:])
            else:
                new_eq=''.join(split_eq[:old_input_index])+new_part+''.join(split_eq[old_input_index+1:])
            if eq_index>0:
                final_new_eq=loops[op_index]+'['+';'.join(equations_under_loops[op_index][:eq_index])+';'+new_eq+';'.join(equations_under_loops[op_index][eq_index+1:])+';];'
            else:
                final_new_eq=loops[op_index]+'['+new_eq+';'.join(equations_under_loops[op_index][eq_index+1:])+';];'
            # print(f'final_new_eq:{final_new_eq}')
            transformed_IR=''.join(row_equations_under_loops[:op_index]) + final_new_eq + ''.join(row_equations_under_loops[op_index+1:])
            transformed_IR_list.append(transformed_IR)
            has_transformation = True
            if 'if_then_else' in new_input:
                if_then_else_transform_idx.append(len(transformed_IR_list)-1)
    if_then_else_transform_idx=list(set(if_then_else_transform_idx))
    # print(f'transformed_IR_list before delete:{transformed_IR_list}')
    if len(if_then_else_transform_idx)>0:
        keep_idx=random.choice(if_then_else_transform_idx)
        # print(f'if_then_else_transform_idx:{if_then_else_transform_idx},len(transformed_IR_list):{len(transformed_IR_list)}')
        remove_transform_IR=[transformed_IR_list[idx] for idx in if_then_else_transform_idx]
        transform_IR_keep=transformed_IR_list[keep_idx]
        for transform_IR in remove_transform_IR:
            transformed_IR_list.remove(transform_IR)
        transformed_IR_list=[transform_IR_keep]+transformed_IR_list
    original_IR_list=[IR]*len(transformed_IR_list)
    return original_IR_list, transformed_IR_list, has_transformation

def apply_additive_split_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    original_IR, transformed_IR_list, has_transformation=apply_multiplicative_split_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops, split_sig='add')
    return original_IR, transformed_IR_list, has_transformation

def sub_normal_loop_max_to_prefix_max(op_index, eq_index, loops, simplified_eqs_under_loops, equations_under_loops, eq_inputs_under_loops, eq_outputs_under_loops, row_equations_under_loops):
    this_loop=loops[op_index]
    this_eq=simplified_eqs_under_loops[op_index][eq_index]
    this_eq_left=this_eq.split('=')[0]
    this_output_list=[item for item in eq_outputs_under_loops[op_index] if this_eq_left in item]
    this_input_list=[item for item in eq_inputs_under_loops[op_index] if this_eq_left not in item]
    this_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(this_input_list[0])
    this_input_subscript_details=generate_subscript_details(this_input_subscript_list)
    this_input_superscript=re.findall(r'\^{[a-zA-Z0-9,]*}', this_input_list[0])[0]
    this_output_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(this_output_list[0])
    new_output_subscript_details=[]
    related_idx=''
    related_idx_num=-1
    values_list, keys_list,_,_=split_loops_into_value_and_index([this_loop])
    for item_idx in range(len(this_input_subscript_details)):
        item=this_input_subscript_details[item_idx]
        if item in this_output_subscript_list[0]:
            new_output_subscript_details.append(item)
        else:
            related_idx_num=item_idx
            related_idx=item+'-1'
            new_output_subscript_details.append(related_idx)
    related_idx_value=values_list[0][keys_list[0].index(this_input_subscript_details[related_idx_num])]
    this_output_subscript='_{'+','.join(new_output_subscript_details)+'}'
    output_part='if_then_else('+related_idx+'<0,-inf,'+this_eq_left+this_input_superscript+this_output_subscript+')'
    new_eq=this_eq_left+this_input_superscript+this_input_subscript_list[0]+'=max('+output_part+','+this_input_list[0]+')'
    if eq_index>0:
        new_part=';'.join(equations_under_loops[op_index][:eq_index])+';'+new_eq+';'+';'.join(equations_under_loops[op_index][eq_index+1:])
    else:
        new_part=new_eq+';'+';'.join(equations_under_loops[op_index][eq_index+1:])
    if eq_index<len(equations_under_loops[op_index])-1:
        new_part+=';];'
    else:
        new_part+='];'
    new_row_eqs=row_equations_under_loops[:op_index] + [this_loop+'['+new_part] + row_equations_under_loops[op_index+1:]
    return new_row_eqs, new_part.replace(';];',''), this_loop, related_idx_num, this_eq_left, related_idx_value

def sub_normal_loop_summation_on_exp_to_prefix_summation_on_exp(name_start_idx, op_index, eq_index, loops, simplified_split_eq_right_part, simplified_eq_inputs_under_loops, equations_under_loops, eq_inputs_under_loops, row_equations_under_loops):
    this_loop=loops[op_index]
    simplified_output=simplified_split_eq_right_part[0]
    first_simplified_input_in_eq=simplified_split_eq_right_part[4]
    second_simplified_input_in_eq=simplified_split_eq_right_part[6]
    split_eq_right_part=[]
    first_full_input_in_eq=0
    second_full_input_in_eq=0
    full_output=0
    intermediate_vars, name_start_idx=generate_names(1, name_start_idx)
    for item in simplified_split_eq_right_part:
        if item in simplified_eq_inputs_under_loops[op_index]:
            full_inputs=[full_item for full_item in eq_inputs_under_loops[op_index] if re.sub(r'\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', full_item)==item or re.sub(r'\^{.*?}','', full_item)==item]
            split_eq_right_part.append(full_inputs[0])
            if item==first_simplified_input_in_eq:
                first_full_input_in_eq=full_inputs[0]
            elif item==second_simplified_input_in_eq:
                second_full_input_in_eq=full_inputs[0]
            elif item==simplified_output:
                full_output=full_inputs[0]
        else:
            split_eq_right_part.append(item)
    first_input_superscript=re.findall(r'\^{[a-zA-Z0-9,]*}', first_full_input_in_eq)[0]
    first_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(first_full_input_in_eq)
    first_input_subscript_details=generate_subscript_details(first_input_subscript_list)
    second_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(second_full_input_in_eq)
    output_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(full_output)
    output_subscript_first_details, output_subscript_second_details, second_input_subscript_first_details, second_input_subscript_second_details=[],[],[],[]
    related_idx_second_input, related_output='',''
    for item in first_input_subscript_details:
        if item in second_input_subscript_list[0]:
            second_input_subscript_first_details.append(item)
            second_input_subscript_second_details.append(item)
        elif item not in second_input_subscript_list[0]:
            related_idx_second_input=item+'-1'
            second_input_subscript_first_details.append(item)
            second_input_subscript_second_details.append(related_idx_second_input)
        if item in output_subscript_list[0]:
            output_subscript_first_details.append(item)
            output_subscript_second_details.append(item)
        else:
            related_output=item+'-1'
            output_subscript_first_details.append(item)
            output_subscript_second_details.append(related_output)
    second_input_first_subscript='_{'+','.join(second_input_subscript_first_details)+'}'
    second_input_second_subscript='_{'+','.join(second_input_subscript_second_details)+'}'
    output_first_subscript='_{'+','.join(output_subscript_first_details)+'}'
    output_second_subscript='_{'+','.join(output_subscript_second_details)+'}'
    exp1_part='if_then_else('+related_idx_second_input+'<0,-inf,'+second_simplified_input_in_eq+first_input_superscript+second_input_second_subscript+')'
    exp1='exp('+ exp1_part+'-'\
    + second_simplified_input_in_eq+first_input_superscript+second_input_first_subscript+')'
    exp2='exp(' + first_full_input_in_eq + '-' + second_simplified_input_in_eq + first_input_superscript + second_input_first_subscript + ')'
    output1=intermediate_vars[0]+first_input_superscript+output_first_subscript
    output2='if_then_else('+related_output+'<0,1,'+intermediate_vars[0]+first_input_superscript+output_second_subscript+')'
    new_eq=output1+'='+ output2+'*'+exp1+'+'+exp2
    if eq_index>0:
        new_part=';'.join(equations_under_loops[op_index][:eq_index])+';'+new_eq+';'+';'.join(equations_under_loops[op_index][eq_index+1:])
    else:
        new_part=new_eq+';'+';'.join(equations_under_loops[op_index][eq_index+1:])
    if eq_index<len(equations_under_loops[op_index])-1:
        new_part+=';];'
    else:
        new_part+='];'
    new_row_eqs=row_equations_under_loops[:op_index] + [this_loop+'['+new_part] + row_equations_under_loops[op_index+1:]
    return new_row_eqs, new_part.replace(';];',''), this_loop, exp1, exp2, output1, output2, intermediate_vars[0], output_subscript_first_details

def apply_normal_loop_summation_on_exp_to_prefix_summation_on_exp_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    has_transformation = False
    transformed_IR_list = []
    original_IR_list=[]
    normal_loop_summation_on_exp_to_prefix_summation_on_exp_index_list=judge_normal_loop_summation_on_exp_to_prefix_summation_on_exp_condition(loops, simplified_eqs_under_loops)
    for index in range(len(normal_loop_summation_on_exp_to_prefix_summation_on_exp_index_list)):
        op_index, eq_index, max_op_index, max_eq_index, simplified_split_eq_right_part = normal_loop_summation_on_exp_to_prefix_summation_on_exp_index_list[index]
        # print(f'op_index:{op_index}, eq_index:{eq_index}, max_op_index:{max_op_index}, max_eq_index:{max_eq_index}, eq:{simplified_eqs_under_loops[op_index][eq_index]}, full_eq:{equations_under_loops[op_index][eq_index]}, split:{simplified_split_eq_right_part}')
        _,new_max_eq,max_loop,max_related_idx_num, max_output, max_idx_value=sub_normal_loop_max_to_prefix_max(max_op_index, max_eq_index, loops, simplified_eqs_under_loops, equations_under_loops, eq_inputs_under_loops, eq_outputs_under_loops, row_equations_under_loops)
        # print(f'new_max_eq:{new_max_eq},max_loop:{max_loop},max_op_index:{max_op_index},max_related_idx_num:{max_related_idx_num},max_idx_value:{max_idx_value}')
        _,new_exp_eq, exp_loop,_,_,_,_, intermediate_var,_=sub_normal_loop_summation_on_exp_to_prefix_summation_on_exp(name_start_idx, op_index, eq_index, loops, simplified_split_eq_right_part, simplified_eq_inputs_under_loops, equations_under_loops, eq_inputs_under_loops, row_equations_under_loops)
        # print(f'new_exp_eq:{new_exp_eq},exp_loop:{exp_loop},op_index:{op_index}')
        if max_loop==exp_loop and max_op_index+1==op_index:
            has_transformation=True
            this_loop=loops[op_index]
            simplified_output=simplified_split_eq_right_part[0]
            first_simplified_input_in_eq=simplified_split_eq_right_part[4]
            first_full_input_in_eq=[item for item in eq_inputs_under_loops[op_index] if first_simplified_input_in_eq in item][0]
            full_output=[item for item in eq_outputs_under_loops[op_index] if simplified_output in item][0]
            values_list, keys_list, loop_types_list,_=split_loops_into_value_and_index([this_loop])
            first_input_superscript=re.findall(r'\^{[a-zA-Z0-9,]*}', first_full_input_in_eq)[0]
            first_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(first_full_input_in_eq)
            first_input_subscript_details=generate_subscript_details(first_input_subscript_list)
            output_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(full_output)
            output_subscript_details=[]
            key_index=[]
            has_tx=False
            # print(f'first_input_subscript_details:{first_input_subscript_details}')
            for item in first_input_subscript_details:
                if item in output_subscript_list[0]:
                    output_subscript_details.append(item)
                    if 'tx' in item:
                        key_index.append(keys_list[0].index('tx'))
                        has_tx=True
                    else:
                        key_index.append(keys_list[0].index(item))
                else:
                    value_idx=keys_list[0].index(item)
                    output_subscript_details.append(str(values_list[0][value_idx]-1))
            output_subscript='_{'+','.join(output_subscript_details)+'}'
            if has_tx:
                new_values_list=[values_list[0][idx] for idx in key_index]
                new_keys_list=[keys_list[0][idx] for idx in key_index]
                new_loop_types_list=[loop_types_list[0][idx] for idx in key_index]
            else:
                new_values_list=[values_list[0][idx] for idx in key_index]
                new_keys_list=['tx']+[keys_list[0][idx] for idx in key_index[1:]]
                new_loop_types_list=['B']+[loop_types_list[0][idx] for idx in key_index[1:]]
            new_loop_notation=curate_loops(new_values_list, new_keys_list, new_loop_types_list)
            # print(f'new_loop_notation:{new_loop_notation},has_tx:{has_tx}')
            new_part=max_loop+'['+new_max_eq+';'+new_exp_eq+';];'
            new_part+=new_loop_notation+'['+simplified_output+first_input_superscript+output_subscript_list[0]+'='+intermediate_var+first_input_superscript+output_subscript+';];'
            transformed_IR=''.join(row_equations_under_loops[:max_op_index]) + new_part
            after_IRs=''.join(row_equations_under_loops[op_index+1:])
            if max_output in after_IRs:
                after_IRs=replace_prefix_output_in_after_IRs(max_output, after_IRs,max_related_idx_num, max_idx_value)
            transformed_IR+=after_IRs
            transformed_IR_list.append(transformed_IR)
            original_IR_list.append(IR)
        # print(f'row_eq:{transformed_IR}')
    return original_IR_list, transformed_IR_list, has_transformation

def apply_online_softmax_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    pre_index_list=judge_normal_loop_summation_on_exp_to_prefix_summation_on_exp_condition(loops, simplified_eqs_under_loops)
    if len(pre_index_list)==0:
        return [IR], [''], False
    else:
        transformed_IR_list=[]
        original_IR_list=[]
        has_transformation=False
        for pre_index in range(len(pre_index_list)):
            exp_op_index, exp_eq_index, max_op_index, max_eq_index, simplified_split_exp_eq_right_part = pre_index_list[pre_index]
            online_softmax_index_list=judge_online_softmax_condition(exp_op_index, exp_eq_index, simplified_split_exp_eq_right_part, loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
            for index in range(len(online_softmax_index_list)):
                op_index, eq_index, simplified_split_eq_right_part, simplified_output= online_softmax_index_list[index]
                # print(f'op_index:{op_index}, eq_index:{eq_index}, eq:{simplified_eqs_under_loops[op_index][eq_index]}, full_eq:{equations_under_loops[op_index][eq_index]}, split:{simplified_split_eq_right_part}')
                _, new_max_eq, max_loop,max_related_idx_num,max_output,max_idx_value=sub_normal_loop_max_to_prefix_max(max_op_index, max_eq_index, loops, simplified_eqs_under_loops, equations_under_loops, eq_inputs_under_loops, eq_outputs_under_loops, row_equations_under_loops)
                # print(f'new_max_eq:{new_max_eq}, max_loop:{max_loop}')
                _, new_exp_eq, exp_loop, _, _, _, _, intermediate_var,_=sub_normal_loop_summation_on_exp_to_prefix_summation_on_exp(name_start_idx, exp_op_index, exp_eq_index, loops, simplified_split_exp_eq_right_part, simplified_eq_inputs_under_loops, equations_under_loops, eq_inputs_under_loops, row_equations_under_loops)
                # print(f'new_exp_eq:{new_exp_eq}, exp_loop:{exp_loop}')
                if max_loop==exp_loop and max_op_index==exp_op_index-1:
                    has_transformation = True
                    first_new_eq=max_loop+'['+new_max_eq+';'+new_exp_eq+';];'
                    # print(f'first_new_eq:{first_new_eq}')
                    new_row_eqs=row_equations_under_loops[:max_op_index] + [first_new_eq] + row_equations_under_loops[exp_op_index+1:]
                    # print(f'new_row_eqs:{new_row_eqs}')
                    first_simplified_input, second_simplified_input, third_simplified_input=simplified_split_eq_right_part[2], simplified_split_eq_right_part[4], simplified_split_eq_right_part[7]
                    first_full_input_in_eq=[item for item in eq_inputs_under_loops[op_index] if first_simplified_input in item][0]
                    second_full_input_in_eq=[item for item in eq_inputs_under_loops[op_index] if second_simplified_input in item][0]
                    third_full_input_in_eq=[item for item in eq_inputs_under_loops[op_index] if third_simplified_input in item][0]
                    full_output=[item for item in eq_outputs_under_loops[op_index] if simplified_output in item][0]
                    first_input_superscript=re.findall(r'\^{[a-zA-Z0-9,]*}', first_full_input_in_eq)[0]
                    first_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(first_full_input_in_eq)
                    second_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(second_full_input_in_eq)
                    third_input_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(third_full_input_in_eq)
                    first_input_subscript_details=generate_subscript_details(first_input_subscript_list)
                    second_input_subscript_details, third_input_subscript_details=[],[]
                    values_list, keys_list, _,_=split_loops_into_value_and_index([loops[op_index]])
                    for item in first_input_subscript_details:
                        if item in second_input_subscript_list[0]:
                            second_input_subscript_details.append(item)
                        else:
                            key_index=keys_list[0].index('tx') if 'tx' in item else keys_list[0].index(item)
                            second_input_subscript_details.append(str(values_list[0][key_index]-1))
                        if item in third_input_subscript_list[0]:
                            third_input_subscript_details.append(item)
                        else:
                            key_index=keys_list[0].index('tx') if 'tx' in item else keys_list[0].index(item)
                            third_input_subscript_details.append(str(values_list[0][key_index]-1))
                    second_input_subscript='_{'+','.join(second_input_subscript_details)+'}'
                    third_input_subscript='_{'+','.join(third_input_subscript_details)+'}'
                    intermediate_full_output=intermediate_var+first_input_superscript+third_input_subscript
                    second_new_eq=full_output+'=exp('+first_full_input_in_eq+'-'\
                    +second_simplified_input+first_input_superscript+second_input_subscript+')/'\
                    +intermediate_full_output
                    if eq_index>0:
                        new_part=loops[op_index]+'['+';'.join(equations_under_loops[op_index][:eq_index])+';'+second_new_eq+';'+';'.join(equations_under_loops[op_index][eq_index+1:])
                    else:
                        new_part=loops[op_index]+'['+second_new_eq+';'+';'.join(equations_under_loops[op_index][eq_index+1:])
                    if eq_index<len(equations_under_loops[op_index])-1:
                        new_part+=';];'
                    else:
                        new_part+='];'
                    transformed_IR=''.join(new_row_eqs[:op_index-1])+new_part
                    after_IRs=''.join(new_row_eqs[op_index:]).replace(third_full_input_in_eq, intermediate_full_output)
                    if max_output in after_IRs:
                        after_IRs=replace_prefix_output_in_after_IRs(max_output, after_IRs,max_related_idx_num, max_idx_value)
                    transformed_IR+=after_IRs
                    transformed_IR_list.append(transformed_IR)
                    original_IR_list.append(IR)
            return original_IR_list, transformed_IR_list, has_transformation

def apply_flashattention_wo_tiling_to_IR(IR, input_output_name, name_start_idx, row_equations_under_loops, loops, equations_under_loops, eq_outputs_under_loops, eq_inputs_under_loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops):
    transformed_IR_list=[]
    original_IR_list=[]
    has_transformation=False
    pre_pre_index_list=judge_normal_loop_summation_on_exp_to_prefix_summation_on_exp_condition(loops, simplified_eqs_under_loops)
    for pre_pre_index in range(len(pre_pre_index_list)):
        exp_op_index, exp_eq_index, max_op_index, max_eq_index, simplified_split_exp_eq_right_part = pre_pre_index_list[pre_pre_index]
        online_softmax_index_list=judge_online_softmax_condition(exp_op_index, exp_eq_index, simplified_split_exp_eq_right_part, loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
        for pre_index in range(len(online_softmax_index_list)):
            div_op_index, div_eq_index, simplified_split_div_eq_right_part, div_output = online_softmax_index_list[pre_index]
            flashattention_wo_tiling_index_list=judge_flashattention_wo_tiling_condition(div_op_index, div_output, loops, simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops)
            for index in range(len(flashattention_wo_tiling_index_list)):
                op_index, eq_index, simplified_split_eq_right_part = flashattention_wo_tiling_index_list[index]
                # print(f'op_index:{op_index}, eq_index:{eq_index}, eq:{simplified_eqs_under_loops[op_index][eq_index]}, full_eq:{equations_under_loops[op_index][eq_index]}, split:{simplified_split_eq_right_part}')
                _, new_max_eq, max_loop,max_related_idx_num,max_output,max_idx_value=sub_normal_loop_max_to_prefix_max(max_op_index, max_eq_index, loops, simplified_eqs_under_loops, equations_under_loops, eq_inputs_under_loops, eq_outputs_under_loops, row_equations_under_loops)
                print(f'max_output:{max_output}')
                _, new_exp_eq, exp_loop,exp1,exp2, exp_output1, exp_output2, _, before_subscript_details=sub_normal_loop_summation_on_exp_to_prefix_summation_on_exp(name_start_idx, exp_op_index, exp_eq_index, loops, simplified_split_exp_eq_right_part, simplified_eq_inputs_under_loops, equations_under_loops, eq_inputs_under_loops, row_equations_under_loops)
                print(f'exp_output1:{exp_output1}')
                if max_loop==exp_loop and max_op_index==exp_op_index-1 and exp_op_index==div_op_index-1:
                    has_transformation = True
                    simplified_output, simplified_first_input=simplified_split_eq_right_part[0], div_output
                    simplified_second_input=simplified_split_eq_right_part[2] if simplified_split_eq_right_part[2]!=div_output else simplified_split_eq_right_part[4]
                    full_output=[item for item in eq_outputs_under_loops[op_index] if simplified_output in item][0]
                    first_full_input_in_eq=[item for item in eq_inputs_under_loops[op_index] if simplified_first_input in item][0]
                    second_full_input_in_eq=[item for item in eq_inputs_under_loops[op_index] if simplified_second_input in item][0]
                    first_input_superscript=re.findall(r'\^{[a-zA-Z0-9,]*}', first_full_input_in_eq)[0]
                    first_input_subscript_list, _= find_subscripts_of_input_output_and_simplified_version(first_full_input_in_eq)
                    full_output_subscript_list, _=find_subscripts_of_input_output_and_simplified_version(full_output)
                    first_input_subscript_details=generate_subscript_details(first_input_subscript_list)
                    full_output_subscript_details=generate_subscript_details(full_output_subscript_list)
                    full_output_subscript_details1, full_output_subscript_details2, full_output_subscript_details3=full_output_subscript_details.copy(), full_output_subscript_details.copy(), full_output_subscript_details.copy()
                    print(f'before_subscript_details:{before_subscript_details}')
                    values_list, keys_list, loop_types_list,_=split_loops_into_value_and_index([loops[op_index]])
                    first_key_index_list=[]
                    second_key_index_list=[]
                    has_tx=False
                    for item in list(set(full_output_subscript_details+first_input_subscript_details)):
                        if item in full_output_subscript_list[0]:
                            if 'tx' in item:
                                has_tx=True
                                temp_idx=[keys_list[0].index('tx')]
                            else:
                                temp_idx=[keys_list[0].index(item)]
                            if item not in first_input_subscript_list[0]:
                                first_key_index_list.extend(temp_idx)
                                second_key_index_list.extend(temp_idx)
                            else:
                                second_key_index_list.extend(temp_idx)
                        else:
                            full_output_subscript_details1.append(item)
                            full_output_subscript_details2.append(item+'-1')
                            temp_index=keys_list[0].index('tx') if 'tx' in item else keys_list[0].index(item)
                            full_output_subscript_details3.append(str(values_list[0][temp_index]-1))
                    print(f'first_key_index_list:{first_key_index_list}')
                    intermediate_keys=generate_idx_names(len(first_key_index_list), len(keys_list[0])+1)
                    full_output_subscript_details1[-2],full_output_subscript_details2[-2]=intermediate_keys[0],intermediate_keys[0]
                    full_output_subscript_details1[-1],full_output_subscript_details2[-1]=before_subscript_details[-1],before_subscript_details[-1]+'-1'
                    related_idx=before_subscript_details[-1]+'-1'
                    second_full_input_in_eq_subscript=re.findall(r'_{.*?}', second_full_input_in_eq)[0]
                    second_full_input_in_eq_subscript_details=second_full_input_in_eq_subscript.replace('_{','').replace('}','').split(',')
                    second_full_input_in_eq_subscript_details[-2]=before_subscript_details[-1]
                    second_full_input_in_eq_subscript_details[-1]=intermediate_keys[0]
                    new_second_full_input_in_eq_subscript='_{'+','.join(second_full_input_in_eq_subscript_details)+'}'
                    second_full_input_in_eq=second_full_input_in_eq.replace(second_full_input_in_eq_subscript,new_second_full_input_in_eq_subscript)
                    full_output_subscript1='_{'+','.join(full_output_subscript_details1)+'}'
                    full_output_subscript2='_{'+','.join(full_output_subscript_details2)+'}'
                    full_output_subscript3='_{'+','.join(full_output_subscript_details3)+'}'
                    name_start_idx+=1
                    intermediate_vars, name_start_idx=generate_names(1, name_start_idx)
                    intermediate_part='if_then_else('+related_idx+'<0,1,'+intermediate_vars[0]+first_input_superscript+full_output_subscript2+')'
                    new_eq=intermediate_vars[0]+first_input_superscript+full_output_subscript1+'='\
                    +intermediate_part+'*'+exp_output2+'*'\
                    +exp1+'/'+exp_output1+'+'+exp2+'/'+exp_output1+'*'+second_full_input_in_eq
                    print(f'second_full_input_in_eq:{second_full_input_in_eq}\nnew_eq:{new_eq}')
                    first_values_list=[values_list[0][idx] for idx in first_key_index_list]
                    first_keys_list=[intermediate_keys[idx] for idx in range(len(first_key_index_list))]
                    first_loop_types_list=[loop_types_list[0][idx] for idx in first_key_index_list]
                    if has_tx:
                        second_values_list=[values_list[0][idx] for idx in second_key_index_list]
                        second_keys_list=[keys_list[0][idx] for idx in second_key_index_list]
                        second_loop_types_list=[loop_types_list[0][idx] for idx in second_key_index_list]
                    else:
                        second_values_list=[values_list[0][idx] for idx in second_key_index_list]
                        second_keys_list=['tx']+[keys_list[0][idx] for idx in second_key_index_list[1:]]
                        second_loop_types_list=['B']+[loop_types_list[0][idx] for idx in second_key_index_list[1:]]
                    first_loop_notation=curate_loops(first_values_list, first_keys_list, first_loop_types_list)
                    second_loop_notation=curate_loops(second_values_list, second_keys_list, second_loop_types_list)
                    new_part=max_loop+'['+new_max_eq+';'+new_exp_eq+';'+first_loop_notation\
                    +'['+new_eq+';];];'
                    new_part+=second_loop_notation+'['+full_output+'='+intermediate_vars[0]+first_input_superscript+full_output_subscript3+';];'
                    # print(f'new_part:{new_part}')
                    full_max_output=new_max_eq.split('=')[0]
                    new_part=renew_intermediate_var(new_part, full_max_output)
                    new_part=renew_intermediate_var(new_part, exp_output1)
                    # print(f'new_part:{new_part}')
                    transformed_IR=''.join(row_equations_under_loops[:max_op_index]) + new_part + ''.join(row_equations_under_loops[op_index+1:])
                    transformed_IR_list.append(transformed_IR)
                    original_IR_list.append(IR)
                    # print(f'transformed_IR:{transformed_IR}')
    return original_IR_list, transformed_IR_list, has_transformation

import re
import random
import math

op_name=['exp', 'max', 'min', 'log', 'abs', 'sqrt', 'erf', 'if_then_else', 'e']
basic_ops=['+', '-', '*', '/', '%', '=', '<', '>', '!', '&', '|']

def generate_names(n, start=0):
    # Define the alphabet excluding "L", "P", "V", "B", and "U"
    alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['L', 'P', 'V', 'B', 'U', 'T']]
    num_letters = len(alphabet)
    names = []
    # Generate names
    for i in range(start, start+n):
        name = ""
        index = i
        while True:
            name = alphabet[index % num_letters] + name
            index = index // num_letters - 1
            if index < 0:
                break
        # Make only the first letter capital
        name = name[0].upper() + name[1:].lower()
        names.append(name)
    return names, start + n

def generate_idx_names(n, start=0):
    # Define the alphabet excluding "L", "P", "V", "B", and "U"
    alphabet_lower = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) not in ['t', 'b', 'v', 'x', 'y', 'z', 'e']]
    num_letters = len(alphabet_lower)
    names = []
    # Generate names
    for i in range(start, start+n):
        name = ""
        index = i
        while True:
            name = alphabet_lower[index % num_letters] + name
            index = index // num_letters - 1
            if index < 0:
                break
        names.append(name)
    return names

def split_IR_to_equations(IR):
    """
    Split the IR into individual equations.
    """
    row_equations_under_loops = re.findall(r'.*?];', IR, re.DOTALL)
    loops=[]
    equations_under_loops = []
    eq_outputs_under_loops = []
    eq_inputs_under_loops = []
    simplified_eqs_under_loops = []
    simplified_eq_outputs_under_loops=[]
    simplified_eq_inputs_under_loops=[]
    for row_eq_under_loops in row_equations_under_loops:
        parts = row_eq_under_loops.split('[', 1)
        #loops
        loops.append(parts[0])
        #eqs
        equations_under_loops.append(parts[1][:-2].split(';')[:-1])
        eq_outputs=[]
        eq_inputs=[]
        #simplified eqs
        simplified_eqs = []
        simplified_eq_outputs = []
        simplified_eq_inputs = []
        for eq in equations_under_loops[-1]:
            split_eq=eq.split('=')
            left_eq='='.join(split_eq[1:])
            # this_output_list=re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}_{.*?}',split_eq[0])+re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}_{.*?{.*?}}',split_eq[0])+re.findall(r',([a-zA-Z]+\^{[a-zA-Z0-9,]*}_{.*?})}',split_eq[0])+re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}_{.*?{.*?}}',split_eq[0])+re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!_{.*?})',split_eq[0])
            # this_input_list=re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}_{.*?}',left_eq)+re.findall(r'{([a-zA-Z]+\^{[a-zA-Z0-9,]*}_{.*?})}',left_eq)+re.findall(r',([a-zA-Z]+\^{[a-zA-Z0-9,]*}_{.*?})}',left_eq)+re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}_{.*?{.*?}}',left_eq)+re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!_{.*?})',left_eq)
            this_output_list=re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',split_eq[0])+re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',split_eq[0])
            this_input_list=re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',left_eq)+re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',left_eq)
            this_output_list=[item for item in this_output_list if item.count('{')==item.count('}') and '+' not in re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', item) and '*' not in re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', item)]
            this_input_list=[item for item in this_input_list if item.count('{')==item.count('}') and '+' not in re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', item) and '*' not in re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', item)]
            eq_outputs.extend(this_output_list)
            eq_inputs.extend(this_input_list)
            #cleaned eq
            cleaned_eq = re.sub(r'\^{[a-zA-Z0-9,]*}', '', eq)
            cleaned_eq = re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}', '', cleaned_eq).replace('}','')
            simplified_eqs.append(cleaned_eq.strip())
            cleaned_eq = cleaned_eq.replace('exp', '').replace('log', '').replace('sqrt', '').replace('abs', '').replace('min', '').replace('max', '').replace('erf', '').replace('if_then_else', '').replace('-inf','').replace('inf', '')
            cleaned_inputs=[re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','',re.sub(r'\^{[a-zA-Z0-9,]*}', '', item)).replace('}','') for item in this_input_list]
            cleaned_outputs=[re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','',re.sub(r'\^{[a-zA-Z0-9,]*}', '', item)).replace('}','') for item in this_output_list]
            simplified_eq_outputs.extend(cleaned_outputs)
            simplified_eq_inputs.extend(cleaned_inputs)
        #outputs and inputs
        eq_outputs_under_loops.append(list(set(eq_outputs)))
        eq_inputs_under_loops.append(list(set(eq_inputs)))
        simplified_eqs_under_loops.append(simplified_eqs)
        #inputs and outputs
        simplified_eq_outputs_under_loops.append(list(set(simplified_eq_outputs)))
        simplified_eq_inputs_under_loops.append(list(set(simplified_eq_inputs)))
    return row_equations_under_loops, loops, equations_under_loops,eq_outputs_under_loops,eq_inputs_under_loops,simplified_eqs_under_loops, simplified_eq_outputs_under_loops, simplified_eq_inputs_under_loops

def split_transformed_IR_to_equations(IR):
    """
    Split the IR into individual equations.
    """
    row_equations_under_loops = re.findall(r'.*?];', IR, re.DOTALL)
    loops=[]
    for row_eq_under_loops in row_equations_under_loops:
        parts = row_eq_under_loops.split('[', 1)
        #loops
        loops.append(parts[0])
    return row_equations_under_loops, loops


def split_loops_into_value_and_index(loops_list):
    values_list = []
    keys_list = []
    loop_type_list=[]
    len_list=[]
    for loops in loops_list:
        values_str = re.findall(r'\^{(\d+)}', loops)
        values=[int(value) for value in values_str]
        keys = re.findall(r'_\{(\w+)=', loops)
        loop_type = re.findall(r'([A-Z])\^', loops)
        values_list.append(values)
        keys_list.append(keys)
        loop_type_list.append(loop_type)
        len_list.append(len(values))
    return values_list, keys_list, loop_type_list, len_list

def check_if_any_special_function_in_equations(expr):
    if (re.sub(r'abs\(.*?\)', '', expr) == expr or re.sub(r'abs\(.*?\)', '', expr) =='') and \
    (re.sub(r'sqrt\(.*?\)', '', expr) == expr or re.sub(r'sqrt\(.*?\)', '', expr) =='') and \
    (re.sub(r'min\(.*?\)', '', expr) == expr or re.sub(r'min\(.*?\)', '', expr) == '') and \
    (re.sub(r'max\(.*?\)', '', expr) == expr or re.sub(r'max\(.*?\)', '', expr) == '') and \
    (re.sub(r'exp\(.*?\)', '', expr) == expr or re.sub(r'exp\(.*?\)', '', expr) == '') and \
    (re.sub(r'log\(.*?\)', '', expr) == expr or re.sub(r'log\(.*?\)', '', expr) == '') and \
    (re.sub(r'if_then_else\(.*?\)', '', expr) == expr or re.sub(r'if_then_else\(.*?\)', '', expr) == ''):
        return False
    return True

def transform_from_original_simplified_expr_to_original_expr(simplified_expr, simplified_var_list, var_list):
    simplified_expr_list=re.findall(r'[a-zA-Z]+|[^a-zA-Z]',simplified_expr)
    expr_list=[]
    scripts=[]
    # print(f'simplified_expr:{simplified_expr},simplified_expr_list:{simplified_expr_list}')
    # print(f'var_list:{var_list}, simplified_var_list:{simplified_var_list}')
    for item in simplified_expr_list:
        if item in simplified_var_list and not item.islower():
            item_index=[i for i in range(len(var_list)) if var_list[i].startswith(item+'^')]
            var_item=var_list[item_index[0]]
            expr_list.append(var_item)
            scripts.append(var_item.replace(item,''))
        else:
            expr_list.append(item)
    expr=''.join(expr_list)
    return expr, scripts

def transfrom_from_original_simpified_expr_to_modified_expr(simplified_expr, simplified_eq_inputs, simplified_eq_outputs, eq_inputs, output_name, output_subscript,output_superscript, inputs_name_list,intermediate_names,inputs_superscripts_list,inputs_subscripts_list):
    simplified_expr_list=re.findall(r'[a-zA-Z]+|[^a-zA-Z]',simplified_expr)
    simplified_inputs_name_list=[item[:item.index('^')] for item in inputs_name_list]
    expr_list=[]
    for item in simplified_expr_list:
        if item in simplified_eq_inputs and not item.islower():
            if item in simplified_inputs_name_list:
                idx= simplified_inputs_name_list.index(item)
                var_item=intermediate_names[idx]+inputs_superscripts_list[idx]+inputs_subscripts_list[idx]
            elif item in simplified_eq_outputs:
                var_item=output_name+output_superscript+output_subscript
            else:
                item_index=[i for i in range(len(eq_inputs)) if eq_inputs[i].startswith(item+'^')]
                var_item=eq_inputs[item_index[0]]
            expr_list.append(var_item)
        else:
            expr_list.append(item)
    expr=''.join(expr_list)
    return expr

def generate_expression_splitting_intermediate_scripts(scripts, output_superscript, expr):
    subscript_lists=[]
    simplified_expr=re.sub(r'\^{.*?}', '', expr).replace('if_then_else','')
    temp_lower_index=re.findall(rf'\b[a-z]+\b',simplified_expr)
    lower_index= [item for item in temp_lower_index if item not in op_name]
    for item in scripts:
        subscript = re.sub(r'\^\{.*?\}', '', item).replace('{', '').replace('}', '').replace('_', '')
        if subscript != '':
            subscript_lists.extend(subscript.split(','))
    # print(f'subscript_lists:{subscript_lists}')
    subscript_lists=list(set(subscript_lists+lower_index))
    if len(subscript_lists)>0:
        return output_superscript+'_{'+','.join(subscript_lists)+'}', True
    else:
        return output_superscript, False

def curate_loops(values, keys, loops):
    loop_notation=''
    for i in range(len(values)):
        loop_notation+=loops[i]+'^{'+str(values[i])+'}_{'+keys[i]+'=0}'
    if loop_notation=='':
        loop_notation='B^{1}_{tx=0}'
    return loop_notation

def rewrite_subscript_for_cancat(full_subscript, diff_last_keys, diff_value_index, last_values_list):
    rewritten_subscript_list = []
    for full_subscript_index in range(len(full_subscript)):
        item=full_subscript[full_subscript_index]
        if item in diff_last_keys:
            idx_in_diff_last_keys= diff_last_keys.index(item)
            value_in_diff_value_index = diff_value_index[idx_in_diff_last_keys]
            if full_subscript_index<len(full_subscript)-1 and full_subscript[full_subscript_index+1]=='*':
                rewritten_subscript_list.append('('+item+'+'+str(last_values_list[value_in_diff_value_index])+')')
            else:
                rewritten_subscript_list.append(item+'+'+str(last_values_list[value_in_diff_value_index]))
        else:
            rewritten_subscript_list.append(item)
    rewritten_subscript = ''.join(rewritten_subscript_list)
    return rewritten_subscript

def find_different_vars_scripts(keys_list, diff_value_index, last_inputs, this_inputs, last_right_simplified_eq, this_right_simplified_eq, values_list):
    diff_last_keys=[keys_list[idx] for idx in diff_value_index]
    num_cancat_inputs=0
    last_inputs_name_list=[]
    this_inputs_name_list=[]
    inputs_superscripts_list=[]
    inputs_subscripts_list=[]
    resubscript_list=[]
    split_last_right_simplified_eq = re.findall(r'[a-zA-Z]+|[^a-zA-Z]', last_right_simplified_eq)
    split_this_right_simplified_eq = re.findall(r'[a-zA-Z]+|[^a-zA-Z]', this_right_simplified_eq)
    # print(f'split_last_right_simplified_eq:{split_last_right_simplified_eq}, split_this_right_simplified_eq:{split_this_right_simplified_eq}')
    for input_index in range(len(last_inputs)):
        last_input=last_inputs[input_index]
        simplified_last_input = last_input[:last_input.index('^')]
        raw_subscript_list=re.findall(r'\_\{.*?\}',last_input)
        raw_subscript=raw_subscript_list[0] if len(raw_subscript_list)>0 else ''
        subscript=re.findall('[a-z]+', raw_subscript)
        if (len(subscript)>len(diff_last_keys) and len(list(set(subscript) - set(diff_last_keys))) < len(list(set(subscript)))) or \
            (len(subscript)<=len(diff_last_keys) and len(list(set(diff_last_keys) - set(subscript))) < len(list(set(diff_last_keys)))):
            num_cancat_inputs+=1
            last_inputs_name_list.append(last_input)
            in_eq_index=split_last_right_simplified_eq.index(simplified_last_input)
            simplified_this_input=split_this_right_simplified_eq[in_eq_index]
            this_inputs_name=[item for item in this_inputs if item.startswith(simplified_this_input+'^')]
            this_inputs_name_list.append(this_inputs_name[0])
            inputs_superscripts_list.append(re.findall(r'\^\{.*?\}',last_input)[0])
            inputs_subscripts_list.append(raw_subscript)
            full_subscript=re.findall(r'[a-zA-Z]+|\d+|[^a-zA-Z]', raw_subscript)
            rewritten_subscript= rewrite_subscript_for_cancat(full_subscript, diff_last_keys, diff_value_index, values_list)
            resubscript_list.append(rewritten_subscript)
    return num_cancat_inputs, last_inputs_name_list, this_inputs_name_list, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, diff_last_keys, diff_value_index

def find_different_vars_scripts_and_curate_loops_for_tensor_concat_fusion(last_values_list, last_keys_list, this_values_list, last_loop_type_list, last_inputs, this_inputs, last_right_simplified_eq, this_right_simplified_eq, output_subscript):
    diff_value_index=[idx for idx in range(len(last_values_list)) if last_values_list[idx] != this_values_list[idx]]
    output_subscript_list=re.findall(r'[a-z]+', output_subscript)
    if len(diff_value_index) == 0:
        diff_value_index=[0]
    if set(output_subscript_list).intersection(set(diff_value_index)):
        diff_value_index.append(output_subscript_list[0])
    num_cancat_inputs, last_inputs_name_list, this_inputs_name_list, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, diff_last_keys, diff_value_index= find_different_vars_scripts(last_keys_list, diff_value_index, last_inputs, this_inputs, last_right_simplified_eq, this_right_simplified_eq, last_values_list)
    new_values_list=[]
    for index in range(len(last_values_list)):
        if index not in diff_value_index:
            new_values_list.append(last_values_list[index])
        else:
            new_values_list.append(this_values_list[index]+last_values_list[index])
    new_loop_notation=curate_loops(new_values_list, last_keys_list, last_loop_type_list)
    return num_cancat_inputs, last_inputs_name_list, this_inputs_name_list, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, new_loop_notation, diff_last_keys, diff_value_index

def random_split_a_loop(value):
    value1 = random.randint(1, value - 1)
    value2 = value - value1
    return value1, value2

def split_right_eq(right_eq):
    #output: new_var_list, split_eq, lower_case_var_list
    new_var_list=[]
    split_eq=[]
    lower_case_var_list=[]
    left_num, right_num = 0, 0
    has_var=False
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
        #stop condition
        if var_idx==len(right_eq)-1:
            split_eq.append(single_eq)
            if single_eq.islower():
                lower_case_var_list.append(single_eq)
            if '^' in single_eq:
                new_var_list.append(single_eq)
        elif ((var_idx!=len(right_eq)-1 and right_eq[var_idx+1] in basic_ops) or var in basic_ops or single_eq in op_name) and not has_var and left_num==0 and right_num==0:
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

def find_different_vars_scripts_and_curate_loops_for_tensor_split_decouple(axis, values_list, keys_list, loop_type_list, inputs, this_right_simplified_eq):
    value1, value2=random_split_a_loop(values_list[axis])
    # print(f'Splitting axis {axis} with values {values_list[axis]} into {value1} and {value2}')
    new_values_list1 = values_list[:axis]+[value1]+ values_list[axis+1:]
    new_values_list2 = values_list[:axis]+[value2]+ values_list[axis+1:]
    first_two_new_values_list1=new_values_list1.copy()
    first_two_new_values_list2=new_values_list2.copy()
    num_cancat_inputs, inputs_name_list, _, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, _, _ = find_different_vars_scripts(keys_list, [axis], inputs, inputs, this_right_simplified_eq, this_right_simplified_eq, new_values_list1)
    contain_tx=False
    for item in resubscript_list:
        if 'tx' in item:
            contain_tx=True
    if not contain_tx:
        idx_tx=keys_list.index('tx')
        first_two_new_values_list1[idx_tx]=1
        first_two_new_values_list2[idx_tx]=1
    loop_notation1 = curate_loops(new_values_list1, keys_list, loop_type_list)
    loop_notation2 = curate_loops(new_values_list2, keys_list, loop_type_list)
    first_two_loop_notation1 = curate_loops(first_two_new_values_list1, keys_list, loop_type_list)
    first_two_loop_notation2 = curate_loops(first_two_new_values_list2, keys_list, loop_type_list)
    # print(f'resubscript_list:{resubscript_list}')
    return num_cancat_inputs, inputs_name_list, inputs_superscripts_list, inputs_subscripts_list, resubscript_list, loop_notation1, loop_notation2,first_two_loop_notation1,first_two_loop_notation2, new_values_list1

def check_subexpression(pattern, expr, subexpression_info, info, simplified_eq_inputs_under_loops, eq_inputs_under_loops):
    if re.sub(pattern, '',expr)!='' and re.sub(pattern, '',expr)!=expr:
        subexpression_key=re.findall(pattern, expr)[0]
        expr, _=transform_from_original_simplified_expr_to_original_expr(subexpression_key, simplified_eq_inputs_under_loops[info[0]], eq_inputs_under_loops[info[0]])
        if expr not in subexpression_info.keys():
            subexpression_info[expr] = [info]
        else:
            subexpression_info[expr].append(info)
    return subexpression_info

def find_all_subscripts_in_expr(expr):
    subscripts_list = re.findall(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}', expr)
    subscripts_list=[item for item in subscripts_list if item.count('{')==item.count('}')]
    subscripts=[]
    for item in subscripts_list:
        subscripts.extend(re.findall(r'[a-z]+',item))
    subscripts=list(set(subscripts))
    return subscripts

def find_subscripts_of_input_output_and_simplified_version(this_input):
    # if len(re.findall(r'_{.*?{.*?}}', this_input))>0:
    #     this_input_subscript_list = re.findall(r'\_{.*?{.*?}}', this_input)
    #     this_simplified_input=re.sub(r'\^{.*?}\_{.*?{.*?}}','', this_input)
    if len(re.findall(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}', this_input))>0:
        this_input_subscript_list = re.findall(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}', this_input)
        this_simplified_input=re.sub(r'\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','', this_input)
    else:
        this_input_subscript_list = []
        this_simplified_input = re.sub(r'\^{.*?}', '', this_input)
    return this_input_subscript_list, this_simplified_input

def generate_subscript_details(this_input_subscript_list):
    subscript_details=re.findall(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',this_input_subscript_list[0][2:-1])
    subscript_details.extend(re.sub(r'[a-zA-Z]+\^{.*?}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}','',this_input_subscript_list[0][2:-1]).split(','))
    if '' in subscript_details:
        subscript_details.remove('')
    return subscript_details

def curate_loops_and_output_subscript_according_to_subscripts(keys_list, values_list, loop_type_list, subscripts):
    same_index=[idx for idx in range(len(keys_list)) if keys_list[idx] in subscripts]
    same_index.sort()
    new_keys_list = [keys_list[idx] for idx in same_index]
    new_values_list = [values_list[idx] for idx in same_index]
    new_loop_type_list = [loop_type_list[idx] for idx in same_index]
    if len(same_index)>0:
        output_subscript='_{'
        for idx in same_index:
            # if keys_list[idx]=='bx':
            #     output_subscript += 'bx*'+str(new_values_list[idx+1])+'+'
            if idx!=same_index[-1]:
                output_subscript += keys_list[idx]+','
            else:
                output_subscript += keys_list[idx]+'}'
        loop_notation = curate_loops(new_values_list, new_keys_list, new_loop_type_list)
        return loop_notation, output_subscript
    else:
        return '', ''
    
def factorization(n):
    factors = [(i, n // i) for i in range(1, math.isqrt(n) + 1) if n % i == 0]
    return factors

def random_shuffule(a):
    while True:
        shuffled = a[:]
        random.shuffle(shuffled)
        if shuffled != a:
            break
    return shuffled

def simplify_expr(expr):
    new_expr=expr.replace('Abs','abs').replace('Max','max').replace('Min','min').replace(' ','')
    new_expr=re.sub(rf'\b(1\.0\*)\b','',new_expr)
    new_expr=re.sub(rf'\b(\*1\.0)\b','',new_expr)
    new_expr=re.sub(rf'\b(1.0)\b','1',new_expr)
    # new_expr=re.sub(r'\(sqrt\((.*?)\)\)', r'sqrt(\1)', new_expr)
    # new_expr=re.sub(r'\(exp\((.*?)\)\)', r'exp(\1)', new_expr)
    # new_expr=re.sub(r'\(log\((.*?)\)\)', r'log(\1)', new_expr)
    # new_expr=re.sub(r'\(min\((.*?)\)\)', r'min(\1)', new_expr)
    # new_expr=re.sub(r'\(max\((.*?)\)\)', r'max(\1)', new_expr)
    # new_expr=re.sub(r'\(erf\((.*?)\)\)', r'erf(\1)', new_expr)
    # new_expr=re.sub(r'\(abs\((.*?)\)\)', r'abs(\1)', new_expr)
    return new_expr

def extract_subsexpression(func_name,expr):
    start_idx = expr.index(func_name)
    left_num, right_num = 0, 0
    for i in range(start_idx, len(expr)):
        if expr[i] == '(':
            left_num += 1
        elif expr[i] == ')':
            right_num += 1
        if left_num == right_num and left_num > 0:
            sub_expr = expr[start_idx:i+1]
            break
    return sub_expr

def split_eq_by_subscript(this_eqs):
    # print(f'this_eqs:{this_eqs}')
    split_eq=[]
    single_elem=''
    start,end=False, False
    start_other_part, end_other_part = True, False
    for var_idx in range(len(this_eqs)-1):
        vars= this_eqs[var_idx:var_idx+2]
        this_var=vars[0]
        next_var=vars[1]
        if vars=='_{' and not start:
            start, end, end_other_part=True, False, True
        elif (next_var=='_' and var_idx<len(this_eqs)-2 and this_eqs[var_idx+2]=='{') and not start and start_other_part and not end_other_part:
            start_other_part, end_other_part = False, True
        elif this_var=='}' and start and not end:
            end, start=True, False
        if (start and not end) or (start_other_part and not end_other_part):
            single_elem+=this_var
        elif end_other_part:
            if this_var!='_' or (this_var=='_' and next_var!='{'):
                single_elem+=this_var
            start_other_part = False
            split_eq.append(single_elem)
            single_elem=''
        elif end:
            single_elem+=this_var
            start,start_other_part = False, True
            split_eq.append(single_elem)
            single_elem=''
    if this_eqs[-1]=='}' and not end and start:
        single_elem+=this_eqs[-1]
        split_eq.append(single_elem)
    else:
        single_elem+=this_eqs[-1]
        if single_elem!='':
            split_eq.append(single_elem)
    return split_eq

def replace_subscript_in_split_eqs(split_this_eqs_by_subscript, original_subscript, new_subscript):
    new_split_eq=[]
    for item_idx in range(len(split_this_eqs_by_subscript)):
        item=split_this_eqs_by_subscript[item_idx]
        if '_' not in item:
            new_split_eq.append(item)
        else:
            split_sub=re.split(r'([a-z]+|[0-9]+)', item)
            for i in range(len(split_sub)):
                if split_sub[i] == original_subscript:
                    split_newsub=re.findall(r'([a-z]+|[0-9]+)', new_subscript)
                    if len(split_newsub)>1 and ((i<len(split_sub)-1 and split_sub[i+1][0] in basic_ops) or (i>0 and split_sub[i-1][0] in basic_ops)):
                        split_sub[i] = '('+new_subscript+')'
                    else:
                        split_sub[i] = new_subscript
            new_split_eq.append(''.join(split_sub))
    return new_split_eq

def replace_comb_subscript_in_split_eqs(split_this_eqs_by_subscript, original_subscript, new_subscript):
    new_split_eq=[]
    for item_idx in range(len(split_this_eqs_by_subscript)):
        item=split_this_eqs_by_subscript[item_idx]
        if '_{' in item:
            split_sub=item.replace('_{','').replace('}','').split(',')
            for i in range(len(split_sub)):
                if split_sub[i] == original_subscript:
                    split_sub[i] = new_subscript
            new_split_eq.append('_{'+','.join(split_sub)+'}')
        else:
            new_split_eq.append(item)
    return new_split_eq

def replace_prefix_output_in_after_IRs(output, after_IRs,related_idx_num, idx_value):
    pattern = rf'{re.escape(output)}\^\{{.*?\}}\_\{{.*?\}}|{re.escape(output)}\^\{{.*?\}}'
    related_outputs=re.findall(pattern,after_IRs)
    for output_item in related_outputs:
        if '_' in output_item:
            subscript = re.findall(r'_{.*?}', output_item)[0]
            subscript_details=subscript.replace('_{','').replace('}','').split(',')
            subscript_details.insert(related_idx_num, str(idx_value-1))
            new_subscript='_{'+','.join(subscript_details)+'}'
            new_output_item=output_item.replace(subscript,new_subscript)
        else:
            subscript_details=[str(idx_value-1)]
            new_subscript='_{'+','.join(subscript_details)+'}'
            output_item=output_item+new_subscript
        after_IRs=after_IRs.replace(output_item, new_output_item)
    return after_IRs

def renew_intermediate_var(new_part, output):
    simplified_output=output[:output.index('^')]
    superscript = re.findall(r'\^{.*?}', output)[0]
    new_superscript=superscript.replace(',g}',',l}')
    new_part=new_part.replace(simplified_output+superscript,simplified_output+new_superscript)
    return new_part

# def check_and_transform_loops(superscript, eq_loop):
#     values_list, keys_list, loop_type_list, _=split_loops_into_value_and_index([eq_loop])
#     subscript = re.sub(r'\^\{.*?\}', '', superscript)
#     keys=re.findall('[a-z]+',subscript)
#     diff_keys=list(set(keys_list[0]) - set(keys))
#     if len(diff_keys) >0:
#         for diff_key in diff_keys:
#             index= keys_list[0].index(diff_key)
#             values_list[0].pop(index)
#             keys_list[0].pop(index)
#             loop_type_list[0].pop(index)
#         loop_notation = curate_loops(values_list[0], keys_list[0], loop_type_list[0])
#         return loop_notation
#     else:
#         return eq_loop
    
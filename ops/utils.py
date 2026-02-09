from torch.fx import symbolic_trace
import inspect
import re
import torch

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

def generate_loop_bind(shape):
    return ['tx']

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
    
def get_torch_dtype(dtype):
    if str(dtype).startswith('torch.'):
        return getattr(torch, str(dtype).split('.')[-1])
    if isinstance(dtype, float):
        return float
    elif isinstance(dtype, int):
        return int

def get_op_input(node):
    op_inputs = []
    for arg in node.args:
        if str(arg)=='T':
            op_inputs.append('op_transpose')
        elif isinstance(arg, tuple) and not isinstance(arg[0], int):
            op_inputs.extend([str(item) for item in arg])
        else:
           op_inputs.append(str(arg)) 
    return op_inputs

def generate_random_tensor(name, inputs_name, inputs_shape, inputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes):
    if name in inputs_name:
        idx = inputs_name.index(name)
        dtype = inputs_dtype[idx]
        shape = inputs_shape[idx]
    elif name in intermediate_names:
        idx = intermediate_names.index(name)
        dtype = intermediate_dtypes[idx]
        shape = intermediate_shapes[idx]
    real_dtype= get_torch_dtype(dtype)
    rand_tensor = torch.randn(*shape).to(get_torch_dtype(real_dtype))
    return rand_tensor

def is_special(s):
    try:
        if ("->" in s) or ("op_transpose" in s) or ("T" in s) or (s=='False') or (s=='True') or (s=='None') or ('(' in s) or (s=='values') or ('[' in s) or ('torch' in s):
            return True
        float(s)
        return True
    except ValueError:
        return False

def rename(ops, inputs_name, outputs_name, intermediate_names, constant_name, params_name, name_start_idx):
    paramsname_constant_mapping = {}
    all_names = inputs_name + intermediate_names + outputs_name + constant_name + params_name
    new_names, name_start_idx = generate_names(len(all_names), start=name_start_idx)
    name_map = {old: new for old, new in zip(all_names, new_names)}
    # replace inputs, outputs, intermediate, constant, and params names
    len_inputs = len(inputs_name)
    len_intermediate = len(intermediate_names)
    len_outputs = len(outputs_name)
    len_constant = len(constant_name)
    new_inputs_name = new_names[:len_inputs]
    new_intermediate_names = new_names[len_inputs:len_inputs+len_intermediate]
    new_outputs_name = new_names[len_inputs+len_intermediate:len_inputs+len_intermediate+len_outputs]
    new_constant_name = new_names[len_inputs+len_intermediate+len_outputs:len_inputs+len_intermediate+len_outputs+len_constant]
    new_params_name = new_names[len_inputs+len_intermediate+len_outputs+len_constant:]
    # print(f'constant_name+params_name:{constant_name+params_name},new_constant_name+new_params_name:{new_constant_name+new_params_name}')
    paramsname_constant_mapping={new: old for old, new in zip(constant_name+params_name, new_constant_name+new_params_name)}
    # rename ops
    for op in ops:
        # print(f'op:{op}')
        op['inputs'] = [name_map[name] if not is_special(name) else name for name in op['inputs']]
        op['outputs'] = [name_map[name] if not is_special(name) else name for name in op['outputs']]
    return ops, new_inputs_name, new_outputs_name, new_intermediate_names, new_constant_name,new_params_name,  name_start_idx, paramsname_constant_mapping

def check_min_max(ops):
    for op_idx in range(len(ops)-1):
        op= ops[op_idx]
        next_op= ops[op_idx+1]
        if (op['name']=='max' or op['name']=='min') and (next_op['name']=='getitem'):
            op['outputs']=next_op['outputs'].copy()
    return ops

def check_size(ops,intermediate_names, intermediate_dtypes,outputs_name, outputs_dtype):
    for op_idx in range(len(ops)-1):
        op= ops[op_idx]
        if str(op['name'])=='size':
            if op['outputs'][0] in intermediate_names:
                idx=intermediate_names.index(op['outputs'][0])
                intermediate_dtypes[idx]=['float']
            elif op['outputs'][0] in outputs_name:
                outputs_dtype=['float']
    return ops,intermediate_dtypes,outputs_dtype


def get_op_info(op_name, node, node_inputs, node_outputs, node_kwargs, module=False, module_node=None):
    op_info={}
    op_info['name']=op_name
    op_info['kwargs']=node_kwargs
    op_info['inputs']=node_inputs
    end=False
    if node.next.op!='output' or (module and module_node.next.op!='output'):
        op_info['outputs'] = node_outputs
    elif node.next.op=='output' or (module and module_node.next.op=='output'):
        op_info['outputs'] = ['output']
        end=True
    return op_info, end

def add_intermediate_info(end, name, node, intermediate_names, intermediate_shapes, intermediate_dtypes):
    if not end:
        intermediate_names.append(name)
        if 'tensor_meta' in node.meta:
            tensor_meta = node.meta['tensor_meta']
            if type(tensor_meta) is torch.fx.passes.shape_prop.TensorMetadata:
                shapes = [tensor_meta.shape]
                dtypes = [tensor_meta.dtype]
            else:
                shapes = [tm.shape for tm in tensor_meta]
                dtypes = [tm.dtype for tm in tensor_meta]
            intermediate_shapes.append(shapes)
            intermediate_dtypes.append(dtypes)
        elif 'type' in node.meta:
            intermediate_shapes.append([torch.Size([])])
            match = re.search(r"(class|type)\s+'([\w\.]+)'", str(node.meta['type']))
            intermediate_dtypes.append([match.group(2)])
        elif str(node.meta)=='{}':
            intermediate_shapes.append([None])
            intermediate_dtypes.append([None])
    return intermediate_names, intermediate_shapes, intermediate_dtypes

def analyze_call_function(ops, node, name_start_idx, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, original_params_name, original_params_shape, original_params_dtype):
    op_inputs=get_op_input(node)
    # print(f'op_inputs:{op_inputs}, node.args:{node.args[0]}')
    match = re.search(r'(function|method)\s+([A-Za-z_]*)', str(node.target))
    op_name = match.group(2)
    op_info, end= get_op_info(op_name, node, op_inputs, [str(node.name)], node.kwargs)
    ops.append(op_info)
    intermediate_names, intermediate_shapes, intermediate_dtypes= add_intermediate_info(end, str(node.name), node, intermediate_names, intermediate_shapes, intermediate_dtypes)
    return ops, name_start_idx, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, original_params_name, original_params_shape, original_params_dtype


def analyze_call_module(ops, traced, node, name_start_idx, inputs_name, inputs_shape, inputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes,params_name, params_shape, params_dtype, original_params_name, original_params_shape, original_params_dtype):
    module_name = node.name
    submod = traced.get_submodule(node.target)
    input_constant_num=0
    call_function_num=0
    try:
        traced_submod = symbolic_trace(submod)
    except Exception as e:
        print(f"Could not trace submodule: {e}")
        print("Retrying with concrete arguments.")
        concrete_args={}
        for name,arg in zip(inspect.signature(submod.forward).parameters.keys(),node.args):
            if not is_special(str(arg)):
                rand_tensor= generate_random_tensor(str(arg), inputs_name, inputs_shape, inputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes)
                concrete_args[name] = rand_tensor
                input_constant_num+=1
            else:
                concrete_args[name] = arg
        traced_submod = symbolic_trace(submod, concrete_args=concrete_args)       
    ph_idx = 0
    name_mapping={}
    # print(f'in original_params_name:{original_params_name}, params_name:{params_name}')
    for sub_node in traced_submod.graph.nodes:
        # print(f"sub_node: {sub_node.op}, name:{node.name}, target: {sub_node.target}, args: {sub_node.args}, kwargs: {sub_node.kwargs}")
        # print("meta:", sub_node.meta)
        if sub_node.op == 'call_function': # op
            call_function_num+=1
            match = re.search(r'(function|method)\s+([A-Za-z_]*)', str(sub_node.target))
            op_name = match.group(2)
            subnode_inputs=[name_mapping[str(arg)] if str(arg) in name_mapping.keys() else str(arg) for arg in sub_node.args]
            if 'norm' in str(sub_node.target):
                norm_params=[name_mapping[str(value)] if str(value) in name_mapping.keys() else str(value) for value in sub_node.kwargs.values()]
                subnode_inputs+=norm_params
            subnode_outputs = [module_name]
            op_info, end = get_op_info(op_name, sub_node, subnode_inputs, subnode_outputs, sub_node.kwargs, True, node)
            ops.append(op_info)
            intermediate_names, intermediate_shapes, intermediate_dtypes= add_intermediate_info(end, module_name, node, intermediate_names, intermediate_shapes, intermediate_dtypes)
        elif sub_node.op == 'get_attr':
            if 'constant'in str(sub_node.target) and input_constant_num>0:
                name_mapping[str(sub_node.target)]=name_mapping['input_'+str(ph_idx-input_constant_num+1)]
                input_constant_num-=1
            elif 'running_' in str(sub_node.target):
                op_name = str(sub_node.target)
                subnode_inputs = [name_mapping[key] for key in name_mapping.keys() if 'input_' in key]
                subnode_outputs = [str(sub_node.target)]
                op_info, end = get_op_info(op_name, sub_node, subnode_inputs, subnode_outputs, sub_node.kwargs, True, node)
                ops.append(op_info)
                intermediate_names, intermediate_shapes, intermediate_dtypes= add_intermediate_info(end, str(sub_node.target), sub_node, intermediate_names, intermediate_shapes, intermediate_dtypes)
            else:
                if len(original_params_name)>0:
                    # subnode_params_name = [str(item) for item in original_params_name]
                    # subnode_params_shape = original_params_shape
                    # subnode_params_dtype = original_params_dtype
                    # params_name.extend(subnode_params_name)
                    # params_shape.extend(subnode_params_shape)
                    # params_dtype.extend(subnode_params_dtype)
                    # original_params_name,original_params_shape,original_params_dtype=[],[],[]
                    # print(f'original_params_name:{original_params_name}')
                    while len(original_params_name)>0:
                        subnode_params_name = original_params_name.pop(0)
                        subnode_params_shape = original_params_shape.pop(0)
                        subnode_params_dtype = original_params_dtype.pop(0)
                        params_name.append(str(subnode_params_name))
                        params_shape.append(subnode_params_shape)
                        params_dtype.append(subnode_params_dtype)
                        if str(node.name)+'.'+str(sub_node.target)==subnode_params_name:
                            break
                    # print(f'after original_params_name:{original_params_name}')
                    # print(f'node.name:{node.name}, sub_node.target:{sub_node.target}')
                    # subnode_params_name = original_params_name.pop(0)
                    # subnode_params_shape = original_params_shape.pop(0)
                    # subnode_params_dtype = original_params_dtype.pop(0)
                    # params_name.append(str(subnode_params_name))
                    # params_shape.append(subnode_params_shape)
                    # params_dtype.append(subnode_params_dtype)
                elif len(original_params_name)==0 and len(params_name)==1 and 'conv' in str(node.name):
                    subnode_params_name = params_name[-1]
                    subnode_params_shape = params_shape[-1]
                    subnode_params_dtype = params_dtype[-1]
                elif len(original_params_name)==0 and len(params_name)==2 and 'gemm' in str(node.name):
                    if 'weight' in str(sub_node.target):
                        subnode_params_name = params_name[-2]
                        subnode_params_shape = params_shape[-2]
                        subnode_params_dtype = params_dtype[-2]
                    elif 'bias' in str(sub_node.target):
                        subnode_params_name = params_name[-1]
                        subnode_params_shape = params_shape[-1]
                        subnode_params_dtype = params_dtype[-1]
                # print(f'sub_node.target:{sub_node.target},name_mapping:{name_mapping}')
                name_mapping[str(sub_node.target)] = str(subnode_params_name)
        elif sub_node.op == 'placeholder':
            submod_name = str(sub_node.target)+'_'+str(ph_idx+1) if '_' not in str(sub_node.target) else str(sub_node.target)
            name_mapping[submod_name]=str(node.args[ph_idx])
            ph_idx += 1
        # print("name_mapping:", name_mapping)
    # print(f'in2 original_params_name:{original_params_name}, params_name:{params_name}')
    return ops, name_start_idx, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, original_params_name, original_params_shape, original_params_dtype


def handle_each_node(traced, placeholder_num, node, ops, name_start_idx, inputs_name, inputs_shape, inputs_dtype, outputs_name, outputs_shape, outputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, constant_name, constant_shape, constant_dtype, constant_value_dict, original_params_name, original_params_shape, original_params_dtype):
    # print(f'node.op:{node.op}, node.target:{node.target}, node.name:{node.name}, node.args:{node.args}')
    if node.op == 'placeholder': #input
        if 'tensor_meta' in node.meta:
            inputs_shape.append(node.meta['tensor_meta'].shape)
            inputs_dtype.append(node.meta['tensor_meta'].dtype)
        elif 'type' in node.meta:
            inputs_shape.append(torch.Size([]))
            match = re.search(r"(class|type)\s+'([\w\.]+)'", str(node.meta['type']))
            inputs_dtype.append(match.group(2))
        if placeholder_num==0:
            inputs_name.append(node.name)
    elif node.op == 'output': #output
        outputs_name.append(node.name)
        outputs_shape.append(node.meta['tensor_meta'].shape)
        outputs_dtype.append(node.meta['tensor_meta'].dtype)
    elif node.op == 'call_function': #op
        ops, name_start_idx, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, original_params_name, original_params_shape, original_params_dtype = analyze_call_function(ops, node, name_start_idx, intermediate_names, intermediate_shapes, intermediate_dtypes,params_name, params_shape, params_dtype, original_params_name, original_params_shape, original_params_dtype)
        # print(f'1 original_params_name:{original_params_name}, params_name: {params_name}')
    elif node.op == 'call_module':
        ops, name_start_idx, intermediate_names, intermediate_shapes, intermediate_dtypes,params_name, params_shape, params_dtype, original_params_name, original_params_shape, original_params_dtype = analyze_call_module(ops, traced, node, name_start_idx,inputs_name, inputs_shape, inputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes,params_name, params_shape, params_dtype, original_params_name, original_params_shape, original_params_dtype)
        # print(f'2 original_params_name:{original_params_name}, params_name: {params_name}')
    elif node.op == 'call_method':
        op_inputs=get_op_input(node)
        op_info, end = get_op_info(str(node.target), node, op_inputs, [str(node.name)], node.kwargs)
        ops.append(op_info)
        intermediate_names, intermediate_shapes, intermediate_dtypes= add_intermediate_info(end, str(node.name), node, intermediate_names, intermediate_shapes, intermediate_dtypes)
    elif node.op == 'get_attr':
        if placeholder_num>0 and 'constant' in str(node.target):
            inputs_name.append(str(node.name))
            placeholder_num-=1
        else:
            # print(f'constant related node: {node}, {node.meta}, {node.target}')
            constant_name.append(str(node.name))
            attr_value = getattr(traced, node.target)
            constant_value_dict[str(node.name)]=attr_value
            # print(f"Constant value for {node.target}: {attr_value}")
            if 'tensor_meta' in node.meta:
                constant_shape.append(node.meta['tensor_meta'].shape)
                constant_dtype.append(node.meta['tensor_meta'].dtype)
            elif 'type' in node.meta:
                constant_shape.append(torch.Size([]))
                match = re.search(r"(class|type)\s+'([\w\.]+)'", str(node.meta['type']))
                constant_dtype.append(match.group(2))
    # print(f'ops:{ops}')
    return placeholder_num, ops, name_start_idx, inputs_name, inputs_shape, inputs_dtype, outputs_name, outputs_shape, outputs_dtype, intermediate_names, intermediate_shapes, intermediate_dtypes, params_name, params_shape, params_dtype, constant_name, constant_shape, constant_dtype, constant_value_dict, original_params_name

#op to IR
def loopup_name(variable_name, inputs_info, outputs_info, intermediate_info, constant_info, params_info):
    inputs_name, inputs_shape, inputs_dtype = inputs_info
    outputs_name, outputs_shape, outputs_dtype = outputs_info
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    constant_name, constant_shape, constant_dtype = constant_info
    params_name, params_shape, params_dtype = params_info
    if variable_name in inputs_name:
        idx = inputs_name.index(variable_name)
        return inputs_shape[idx], str(inputs_dtype[idx])
    elif variable_name in outputs_name:
        idx = outputs_name.index(variable_name)
        return outputs_shape[idx], str(outputs_dtype[idx])
    elif variable_name in intermediate_names:
        idx = intermediate_names.index(variable_name)
        if len(intermediate_shapes[idx])==1:
            return intermediate_shapes[idx][0], str(intermediate_dtypes[idx][0])
        elif len(intermediate_shapes[idx])>1:
            return intermediate_shapes[idx], intermediate_dtypes[idx]
    elif variable_name in constant_name:
        idx = constant_name.index(variable_name)
        return constant_shape[idx], str(constant_dtype[idx])
    elif variable_name in params_name:
        idx = params_name.index(variable_name)
        return params_shape[idx], str(params_dtype[idx])
    elif is_special(variable_name):
        return None, None

def loopup_variables(op_inputs, op_outputs, inputs_info, outputs_info, intermediate_info, constant_info, params_info):
    op_inputs_shape, op_inputs_dtype=[],[]
    op_outputs_shape, op_outputs_dtype=[],[]
    #lookup the inputs and outputs shape and dtype
    for op_input in op_inputs:
        input_shape, input_dtype = loopup_name(op_input, inputs_info, outputs_info, intermediate_info, constant_info, params_info)
        op_inputs_shape.append(input_shape)
        op_inputs_dtype.append(input_dtype)
    for op_output in op_outputs:
        output_shape, output_dtype = loopup_name(op_output, inputs_info, outputs_info, intermediate_info, constant_info, params_info)
        op_outputs_shape.append(output_shape)
        op_outputs_dtype.append(output_dtype)
    return op_inputs_shape, op_inputs_dtype, op_outputs_shape, op_outputs_dtype

def generate_var_IR(var, idx, dtype, shape):
    #dtype mapping
    dtype_mapping = {'UNDEFINED':'undef', 'FLOAT32':'f32', 'UINT8':'u8', 'INT8':'i8', 
                     'UINT16':'u16', 'INT16':'i16', 'INT32':'i32', 'INT64':'i64', 
                     'STRING':'str', 'BOOL':'bool', 'FLOAT16':'f16', 'DOUBLE':'f64',
                     'UINT32':'u32', 'UINT64':'u64', 'COMPLEX64':'c64',
                     'COMPLEX128':'c128', 'BFLOAT16':'bf16',
                     'FLOAT8E4M3FN':'f8e4m3fn', 'FLOAT8E4M3FNUZ':'f8e4m3fnuz',
                     'FLOAT8E5M2':'f8e5m2', 'FLOAT8E5M2FNUZ':'f8e5m2fnu',
                     'UINT4':'u4', 'INT4':'i4'}
    
    if shape is not None:
        dtype_key=dtype.replace('torch.', '').upper()
        if dtype_key=='FLOAT':
            dtype_key='DOUBLE'
        elif dtype_key=='FLOAT64':
            dtype_key='DOUBLE'
        elif dtype_key=='INT':
            dtype_key='INT32'
        superscript=dtype_mapping[dtype_key]+',g'
        if isinstance(dtype, str) and dtype!='None' and len(shape)>0:    
            idx_str = str(idx).replace('[','').replace(']','').replace('\'', '')
            return var+'^{'+superscript+'}_{'+idx_str+'}'
        elif len(shape)==0:
            return var+'^{'+superscript+'}'
    else:
        return var

def generate_partial_input_idx_for_mul_add_sub_truediv(idx, shape, op_outputs_shape):
    partial_input_idx=[]
    start_output=len(op_outputs_shape[0])-len(shape)
    if len(shape)==len(op_outputs_shape[0]):
        for i in range(len(shape)):
            dim_input = shape[i]
            dim_output = op_outputs_shape[0][i]
            if dim_input == dim_output:
                partial_input_idx.append(idx[i])
            elif dim_input == 1:
                partial_input_idx.append('0')
            else:
                partial_input_idx.append(idx[i])
    else:
        shape_str=str(list(shape)).replace('[','').replace(']','')
        output_shape_str=str(list(op_outputs_shape[0])).replace('[','').replace(']','')
        if shape_str in output_shape_str:
            start_idx_in_str= output_shape_str.index(shape_str)
            if output_shape_str[:start_idx_in_str]!='':
                start_idx_in_list=len(eval(output_shape_str[:start_idx_in_str]))
            else:
                start_idx_in_list=len(op_outputs_shape[0])
            if shape_str!='1':
                partial_input_idx+=idx[start_idx_in_list:]
            else:
                partial_input_idx.append('0')
        else:
            for i in range(len(shape)):
                dim_input = shape[i]
                for j in range(start_output, len(op_outputs_shape[0])):
                    dim_output = op_outputs_shape[0][j]
                    if dim_input == dim_output:
                        start_output = j + 1
                        partial_input_idx.append(idx[j])
                        break
                    elif dim_input == 1:
                        start_output = j
                        partial_input_idx.append('0')
                        break
    return partial_input_idx

def generate_input_idx_for_mul_add_sub_truediv(idx, op_inputs_shape, op_outputs_shape):
    # print(f'op_inputs_shape:{op_inputs_shape}, op_outputs_shape:{op_outputs_shape}')
    if op_inputs_shape[0] == op_inputs_shape[1]:
        input_idx=[idx,idx]
    elif op_inputs_shape[0] is not None and op_inputs_shape[1] is not None and len(op_inputs_shape[0])>0 and len(op_inputs_shape[1])>0:
        if op_inputs_shape[0]==op_outputs_shape[0]:
            input_idx=[idx]
            input_idx.append(generate_partial_input_idx_for_mul_add_sub_truediv(idx, op_inputs_shape[1], op_outputs_shape))
        else:
            input_idx=[generate_partial_input_idx_for_mul_add_sub_truediv(idx, op_inputs_shape[0], op_outputs_shape)]
            input_idx.append(idx)
    else:
        if op_inputs_shape[0]==op_outputs_shape[0]:
            input_idx=[idx,None]
        else:
            input_idx=[None,idx]
    return input_idx

def mean_var_idx_and_loops(op_kwargs, op_inputs_shape, op_outputs_shape, idx_prefix):
    if len(op_kwargs)>0 and 'dim' in op_kwargs and op_kwargs['dim'] is not None:
        dim= [op_kwargs['dim']] if isinstance(op_kwargs['dim'], int) else list(op_kwargs['dim'])
        keep_dim = op_kwargs['keepdim'] if 'keepdim' in op_kwargs else False
        input_dim=list(range(len(op_inputs_shape[0])))
        diff_dim = list(set(input_dim) - set(dim))
        selected_diff_dim=diff_dim[0] if len(diff_dim)>0 else 0
        first_loops, second_loops = '', ''
        input_idx = idx_prefix[:selected_diff_dim] + ['tx'] + idx_prefix[selected_diff_dim:]
        first_loops_bound= 'B^{' + str(op_inputs_shape[0][selected_diff_dim]) + '}_{tx=0}'
        for i in range(0, selected_diff_dim):
            first_loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + input_idx[i] + '=0}'
        first_loops += first_loops_bound
        for i in range(selected_diff_dim+1, len(op_inputs_shape[0])):
            first_loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + input_idx[i] + '=0}'
        if len(op_inputs_shape[0])==len(op_outputs_shape[0]):
            output_idx=input_idx
        else:
            output_idx = ['tx']
            for i in range(1,len(diff_dim)):
                output_idx.append(input_idx[diff_dim[i]])
        for i in range(len(output_idx)):
            if 'tx' in output_idx[i]:
                second_loops += first_loops_bound
            else:
                second_loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + output_idx[i] + '=0}'
        div_num=torch.prod(torch.tensor(op_inputs_shape[0])[list(dim)]).item()
        if keep_dim:
            intermediate_idx= output_idx#[input_idx[i] for i in diff_dim]
            intermediate_idx2=[]
            for i in range(len(op_outputs_shape[0])):
                item= op_outputs_shape[0][i]
                if item!=1:
                    intermediate_idx2.append(input_idx[i])
                else:
                    intermediate_idx2.append('0')
            intermediate_shape= torch.Size([op_inputs_shape[0][i] for i in diff_dim])
        else:
            intermediate_idx = output_idx
            intermediate_idx2 = output_idx
            intermediate_shape = op_outputs_shape[0]
        return input_idx, output_idx, first_loops, second_loops, div_num, intermediate_idx, intermediate_idx2, intermediate_shape

    else:
        input_idx=['tx'] + idx_prefix
        output_idx = None
        first_loops='B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
        for i in range(1, len(op_inputs_shape[0])):
            first_loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + input_idx[i] + '=0}'
        second_loops = ''
        div_num=torch.prod(torch.tensor(op_inputs_shape[0])).item()
        if first_loops=='':
            first_loops='B^{1}_{tx=0}'
        if second_loops=='':
            second_loops='B^{1}_{tx=0}'
        return input_idx, output_idx, first_loops, second_loops, div_num, [], [], torch.Size([])

def is_num_in_tuple(num, padding):
    if num in padding:
        padding_idx=[i for i, value in enumerate(padding) if value != num]
        if len(padding_idx) > 0:
            is_Padding = True
        else:
            is_Padding = False
    else:
        padding_idx = list(range(len(padding)))
        is_Padding = True
    return is_Padding, padding_idx

def view_info(long_tensor_shape, short_tensor_shape):
    index_len = len(long_tensor_shape)
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(long_tensor_shape[0]) + idx
    loops = 'B^{' + str(long_tensor_shape[0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(long_tensor_shape[i]) + '}_{' + idx[i] + '=0}'
    if len(long_tensor_shape) == len(short_tensor_shape):
        output_idx = [idx[j] for i, dim_long in enumerate(long_tensor_shape) for j, dim_short in enumerate(short_tensor_shape) if dim_long == dim_short]
    else:
        output_idx=[]
        start_long=0
        for i in range(len(short_tensor_shape)):
            dim_short = short_tensor_shape[i]
            l_value=1
            output_idx_item=''
            for j in range(start_long,len(long_tensor_shape)):
                dim_long = long_tensor_shape[j]
                if dim_long == dim_short:
                    start_long = j + 1
                    output_idx.append(idx[j])
                    break
                elif l_value*dim_long == dim_short and l_value!=1:
                    start_long = j + 1
                    output_idx.append(output_idx_item+idx[j])
                    break
                elif l_value*dim_long != dim_short:
                    output_idx_item+=idx[j]+'*'+str(int(dim_short/dim_long))+'+'
                    l_value *= dim_long
    return idx, loops, output_idx

# #old version
# def create_input_output_IR(idx_lists, name_list, dtype_list, is_split=False, axis=None, num_output=1, input_shape=None, is_reshape=False, different_index=None, long_tensor_shape=None):
#     idx=0
#     IRs=[]
#     for name in name_list:
#         reshape_alphabet=""
#         for split_idx in range(num_output):
#             IR=name+"^{"+str(dtype_list[idx])+",g}"
#             idx_inner=0
#             for i in idx_lists[idx]:
#                 if i>=0:
#                     alphabet=alphabet_lower[i]
#                 else:
#                     alphabet="0"
#                 if is_split:
#                     split_size=input_shape[0][axis]//num_output
#                     if i==axis and split_idx!=0:
#                         alphabet+="+"+str(split_size*split_idx)
#                 elif is_reshape:
#                     if i in different_index and len(different_index)>1:
#                         alphabet+="*"+str(long_tensor_shape[i])
#                         reshape_alphabet+=alphabet
#                         different_index=different_index[1:]
#                         continue
#                     elif i in different_index and len(different_index)==1:
#                         reshape_alphabet+="+"+alphabet
#                         alphabet=reshape_alphabet
#                         different_index=different_index[1:]
#                 if idx_inner==0: 
#                     IR+="_{"
#                 IR+=alphabet
#                 if idx_inner<len(idx_lists[idx])-1:
#                     IR+=","
#                 elif idx_inner==len(idx_lists[idx])-1:
#                     IR+="}"
#                 idx_inner+=1
#             IRs.append(IR)
#         idx+=1
#     return IRs

# def get_input_idx_list(input_shape,reduce_axis_shape, shape_list):
#     position_in_input2=input_shape[1].index(reduce_axis_shape)
#     idx_list_1=list(range(position_in_input2))+list(range(position_in_input2,len(input_shape[0])-1))+[len(shape_list)-1]
#     idx_list_2=list(range(position_in_input2))+[len(shape_list)-1]+list(range(len(input_shape[0])-1,len(shape_list)-1))

#     return [idx_list_1, idx_list_2]

# def replace_disallowed_chars(info: str) -> str:
#     disallowed = set(['t', 'b', 'v', 'x', 'y', 'z'])
#     available = [ch for ch in alphabet_lower if ch not in info]
#     mapping = {}
#     result_chars = []
#     for ch in info:
#         if ch in disallowed:
#             if ch not in mapping:
#                 if available:
#                     mapping[ch] = available.pop(0)
#             result_chars.append(mapping[ch])
#         else:
#             result_chars.append(ch)
#     return "".join(result_chars)

# def input_output_IR_add_index(IR_list, index_list):
#     idx=0
#     IRs=[]
#     for ir in IR_list:
#         index=",".join(index_list[idx])
#         IRs.append(ir+"_{"+index+"}")
#     return IRs
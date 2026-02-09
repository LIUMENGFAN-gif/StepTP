from .utils import *

def matmul_to_IR(op_inputs,op_inputs_shape, op_inputs_dtype, op_outputs,op_outputs_shape, op_outputs_dtype):
    #matmul has two inputs and one output
    index_len = len(op_outputs_shape[0])
    idx=generate_idx_names(index_len, 0)
    idx=generate_loop_bind(op_outputs_shape[0][0])+idx
    input_idx0=idx[:len(op_inputs_shape[0])-1]+[idx[-1]]
    if len(op_inputs_shape[0])+len(op_inputs_shape[1])-1==len(op_outputs_shape[0]):
        input_idx1=[idx[-1]]+idx[len(op_inputs_shape[0])-1:len(op_inputs_shape[0])+len(op_inputs_shape[1])-2]
    else:
        overlap_num=len(op_inputs_shape[0])+len(op_inputs_shape[1])-2-len(op_outputs_shape[0])
        input_idx1=idx[:overlap_num]+[idx[-1]]+idx[len(op_inputs_shape[0])-1:len(op_inputs_shape[0])+len(op_inputs_shape[1])-overlap_num-2]
    var_inputs = [generate_var_IR(op_inputs[0], input_idx0, op_inputs_dtype[0], op_inputs_shape[0]),
                   generate_var_IR(op_inputs[1], input_idx1, op_inputs_dtype[1], op_inputs_shape[1])]
    var_outputs = generate_var_IR(op_outputs[0], idx[:index_len], op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR='B^{'+str(op_outputs_shape[0][0])+'}_{tx=0}'
    for i in range(1,index_len):
        IR+='L^{'+str(op_outputs_shape[0][i])+'}_{'+idx[i]+'=0}'
    IR+='L^{'+str(op_inputs_shape[0][-1])+'}_{'+idx[-1]+'=0}'
    IR+='['+var_outputs+'='+var_outputs+'+'+var_inputs[0]+'*'+var_inputs[1]+';];'
    return IR

def bmm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #bmm has two inputs and one output
    index_len = len(op_outputs_shape[0])
    idx=generate_idx_names(index_len, 0)
    idx=generate_loop_bind(op_outputs_shape[0][0])+idx
    var_inputs = [generate_var_IR(op_inputs[0], idx[:len(op_inputs_shape[0])-1]+[idx[-1]], op_inputs_dtype[0], op_inputs_shape[0]),
                   generate_var_IR(op_inputs[1], [idx[0]]+[idx[-1]]+idx[len(op_inputs_shape[0])-1:len(op_inputs_shape[0])+len(op_inputs_shape[1])-3], op_inputs_dtype[1], op_inputs_shape[1])]
    var_outputs = generate_var_IR(op_outputs[0], idx[:index_len], op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR='B^{'+str(op_outputs_shape[0][0])+'}_{tx=0}'
    for i in range(1,index_len):
        IR+='L^{'+str(op_outputs_shape[0][i])+'}_{'+idx[i]+'=0}'
    IR+='L^{'+str(op_inputs_shape[0][-1])+'}_{'+idx[-1]+'=0}'
    IR+='['+var_outputs+'='+var_outputs+'+'+var_inputs[0]+'*'+var_inputs[1]+';];'
    return IR

def einsum_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #einsum has three inputs and one output
    #inputs: reduce axis, input1, input2
    inputs_idx, output_idx = op_inputs[0].split('->')
    reduce_axis = list(set(inputs_idx.replace(',',''))-set(output_idx))
    index_len=len(set(op_inputs[0].replace('->','').replace(',','')))-1
    idx=generate_idx_names(index_len, 0)
    idx=generate_loop_bind(op_outputs_shape[0][0])+idx
    idx_mapping=dict(zip(output_idx, idx[:len(output_idx)]))|dict(zip(reduce_axis, idx[len(output_idx):]))
    split_inputs_idx = inputs_idx.split(',')
    var_inputs = [generate_var_IR(op_inputs[1], [idx_mapping[item] for item in list(split_inputs_idx[0])], op_inputs_dtype[1], op_inputs_shape[1]),
                   generate_var_IR(op_inputs[2], [idx_mapping[item] for item in list(split_inputs_idx[1])], op_inputs_dtype[2], op_inputs_shape[2])]
    var_outputs = generate_var_IR(op_outputs[0], idx[:len(output_idx)], op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR='B^{'+str(op_outputs_shape[0][0])+'}_{tx=0}'
    for i in range(1, len(op_outputs_shape[0])):
        IR+='L^{'+str(op_outputs_shape[0][i])+'}_{'+idx[i]+'=0}'
    for i in range(len(reduce_axis)):
        shape_idx=split_inputs_idx[0].index(reduce_axis[i])
        IR+='L^{'+str(op_inputs_shape[1][shape_idx])+'}_{'+idx_mapping[reduce_axis[i]]+'=0}'
    IR+='['+var_outputs+'='+var_outputs+'+'+var_inputs[0]+'*'+var_inputs[1]+';];'
    return IR

def softmax_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #exp(x - max(x)) / sum(exp(x - max(x))), one input and one output, kwargs: dim=1
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    dim= op_kwargs['dim']
    if dim<0:
        dim += len(op_inputs_shape[0])
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    if dim!=0:
        idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
        loops = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
        for i in range(1, index_len):
            loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    else:
        idx = idx + generate_loop_bind(op_outputs_shape[0][0])
        loops=''
        for i in range(0, index_len-1):
            loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
        loops += 'B^{' + str(op_outputs_shape[0][-1]) + '}_{tx=0}'
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_names, name_start_idx = generate_names(2, name_start_idx)
    temp_intermediate_shapes = [[torch.Size(op_outputs_shape[0][:dim]+op_outputs_shape[0][dim+1:])]] * 2
    var_intermediate=[generate_var_IR(temp_intermediate_names[0], idx[:dim]+idx[dim+1:], op_inputs_dtype[0], temp_intermediate_shapes[0][0]),
                      generate_var_IR(temp_intermediate_names[1], idx[:dim]+idx[dim+1:], op_inputs_dtype[0], temp_intermediate_shapes[1][0])]
    intermediate_names += temp_intermediate_names
    intermediate_shapes += temp_intermediate_shapes
    intermediate_dtypes += [[op_inputs_dtype[0]]] * 2  
    if loops=='':
        loops='B^{1}_{tx=0}'
    #max
    IR = loops + '[' + var_intermediate[0] + '=' + 'max(' +var_intermediate[0]+','+ var_inputs + ');];'
    #exp sum
    IR += loops + '[' + var_intermediate[1] + '=' + var_intermediate[1] +'+'+ 'exp(' + var_inputs + '-' + var_intermediate[0] + ');];'
    #frac
    IR += loops + '[' + var_outputs + '=exp(' + var_inputs + '-' + var_intermediate[0] + ')/' + var_intermediate[1] + ';];'
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def mul_add_sub_truediv_to_IR(op_name, op_inputs,op_inputs_shape, op_inputs_dtype, op_outputs,op_outputs_shape, op_outputs_dtype):
    #mul has two inputs and one output
    index_len = len(op_outputs_shape[0])
    # print(f'index_len:{index_len}')
    if index_len>0:
        idx = generate_idx_names(index_len-1, 0)
        idx = generate_loop_bind(op_outputs_shape[0][0]) + idx 
        input_idx = generate_input_idx_for_mul_add_sub_truediv(idx, op_inputs_shape, op_outputs_shape)
        # print(f'input_idx:{input_idx}')
        var_inputs = [generate_var_IR(op_inputs[0], input_idx[0], op_inputs_dtype[0], op_inputs_shape[0]),
                    generate_var_IR(op_inputs[1], input_idx[1], op_inputs_dtype[1], op_inputs_shape[1])]
        # print(f'input_idx: {input_idx}')
        var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_dtype[0])
        #loops
        IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
        for i in range(1, index_len):
            IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    elif index_len==0:
        var_inputs = [generate_var_IR(op_inputs[0], [], op_inputs_dtype[0], op_inputs_shape[0]),
                      generate_var_IR(op_inputs[1], [], op_inputs_dtype[1], op_inputs_shape[1])]
        var_outputs = generate_var_IR(op_outputs[0], [], op_outputs_dtype[0], op_outputs_shape[0])
        IR='B^{1}_{tx=0}'
    if op_name == 'mul' or op_name=='multiply':
        IR += '[' + var_outputs + '=' + var_inputs[0] + '*' + var_inputs[1] + ';];'
    elif op_name == 'add':
        IR += '[' + var_outputs + '=' + var_inputs[0] + '+' + var_inputs[1] + ';];'
    elif op_name == 'sub':
        IR += '[' + var_outputs + '=' + var_inputs[0] + '-' + var_inputs[1] + ';];'
    elif op_name == 'truediv':
        IR += '[' + var_outputs + '=' + var_inputs[0] + '/' + var_inputs[1] + ';];'
    return IR

def clamp_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs):
    #clamp has one input and one output, kwargs: min, max
    keys_kwargs = op_kwargs.keys()
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0])+ idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    if 'min' in keys_kwargs and 'max' in keys_kwargs:
        #kwargs has min and max, then min(max(x_i, min_value), max_value)
        min_value = op_kwargs['min']
        max_value = op_kwargs['max']
        IR += '[' + var_outputs + '=' + 'min(max(' + var_inputs + ',' + str(min_value) + '),' + str(max_value) + ');];'
    elif len(op_inputs)==3:
        try:
            min_value = eval(op_inputs[1])
            max_value = eval(op_inputs[2])
            IR+= '[' + var_outputs + '=' + 'min(max(' + var_inputs + ',' + str(min_value) + '),' + str(max_value) + ');];'
        except Exception as e:
            min_var= generate_var_IR(op_inputs[1], [], op_inputs_dtype[1], op_inputs_shape[1])
            max_var= generate_var_IR(op_inputs[2], [], op_inputs_dtype[2], op_inputs_shape[2])
            IR += '[' + var_outputs + '=' + 'min(max(' + var_inputs + ',' + min_var + '),' + max_var + ');];'
    elif 'min' in keys_kwargs:
        #kwargs only has min, then max(input, min_value)
        min_value = op_kwargs['min']
        IR += '[' + var_outputs + '=' + 'max(' + var_inputs + ',' + str(min_value) + ');];'
    elif 'max' in keys_kwargs:
        #kwargs only has max, then min(input, max_value)
        max_value = op_kwargs['max']
        IR += '[' + var_outputs + '=' + 'min(' + var_inputs + ',' + str(max_value) + ');];'    
    return IR

def mean_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #mean has one input and one output, kwargs: dim, keepdim
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    index_len = len(op_inputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
    input_idx, output_idx, first_loops, second_loops, div_num, intermediate_idx,intermediate_idx2, temp_intermediate_shape=mean_var_idx_and_loops(op_kwargs, op_inputs_shape, op_outputs_shape, idx_prefix)
    # print(f'intermediate_idx:{intermediate_idx}')
    var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    var_intermediate = generate_var_IR(temp_intermediate_names[0], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape)
    var_intermediate2 = generate_var_IR(temp_intermediate_names[0], intermediate_idx2, op_inputs_dtype[0], temp_intermediate_shape)
    # print(f'var_intermediate:{var_intermediate}')
    IR= first_loops + '[' + var_intermediate2 + '=' + var_intermediate2 + '+' + var_inputs + ';];'
    IR += second_loops + '[' + var_outputs + '=' + var_intermediate + '/' + str(div_num) + ';];'
    intermediate_names += temp_intermediate_names
    intermediate_shapes += [[temp_intermediate_shape]]
    intermediate_dtypes += [[op_inputs_dtype[0]]]
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def running_var_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #var has two input (input, bias) and one output, kwargs: dim, keepdim
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    keys_kwargs = op_kwargs.keys()
    index_len = len(op_inputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
    input_idx, output_idx, first_loops, second_loops, div_num, intermediate_idx,intermediate_idx2, temp_intermediate_shape=mean_var_idx_and_loops(op_kwargs, op_inputs_shape, op_outputs_shape, idx_prefix)
    var_inputs = [generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0]),
                     generate_var_IR(op_inputs[1], output_idx, op_inputs_dtype[1], op_inputs_shape[1])]
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    var_intermediate = generate_var_IR(temp_intermediate_names[0], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape)
    var_intermediate2 = generate_var_IR(temp_intermediate_names[0], intermediate_idx2, op_inputs_dtype[0], temp_intermediate_shape)
    IR= first_loops + '[' + var_intermediate2 + '=' + var_intermediate2 + '+' + '(' + var_inputs[0] + '-' + var_inputs[1] + ')**2;];'
    if 'unbiased' in keys_kwargs and op_kwargs['unbiased']:
    #unbiased, then divide by n-1
        IR += second_loops + '[' + var_outputs + '=' + var_intermediate + '/' + str(div_num-1) + ';];'
    else:
        #biased, then divide by n
        IR += second_loops + '[' + var_outputs + '=' + var_intermediate + '/' + str(div_num) + ';];'
    intermediate_names += temp_intermediate_names
    intermediate_shapes += [[temp_intermediate_shape]]
    intermediate_dtypes += [[op_inputs_dtype[0]]]
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def diag_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #diag has one input and one output
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], [idx[-1]], op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}L^{' + str(op_outputs_shape[0][1]) + '}_{' + idx[1] + '=0}'
    IR += '[' + var_outputs + '=' + 'if_then_else('+idx[1]+'=='+idx[0]+',' + var_inputs + ',0);];'
    return IR

def getattr_transpose_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #transpose has one input and one output, inputs: (0,1)
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx[::-1], op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + var_inputs + ';];'
    return IR

def triu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #triu has one input and one output
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + 'if_then_else(' + idx[-2] + '<=' + idx[-1] + ',' + var_inputs + ',0);];'
    return IR

def tril_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #tril has one input and one output
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + 'if_then_else(' + idx[-2] + '>=' + idx[-1] + ',' + var_inputs + ',0);];'
    return IR

def relu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #relu has one input and one output
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + 'max(' + var_inputs + ', 0);];'
    return IR

def leaky_relu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs):
    #leaky_relu has one input and one output, kwargs: negative_slope
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    negative_slope = op_kwargs['negative_slope']
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + 'if_then_else(' + var_inputs + '<0,' + str(negative_slope) + '*' + var_inputs + ',' + var_inputs + ');];'
    return IR

def sigmoid_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #sigmoid has one input and one output
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + '1/(1+exp(-' + var_inputs + '));];'
    return IR

def tanh_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, intermediate_info, name_start_idx):
    #tanh has one input and one output
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_names, name_start_idx = generate_names(2, name_start_idx)
    temp_intermediate_shapes = [[op_inputs_shape[0]]] * 2
    var_intermediate=[generate_var_IR(temp_intermediate_names[0], idx, op_inputs_dtype[0], temp_intermediate_shapes[0][0]),
                      generate_var_IR(temp_intermediate_names[1], idx, op_inputs_dtype[0], temp_intermediate_shapes[1][0])]
    intermediate_names += temp_intermediate_names
    intermediate_shapes += temp_intermediate_shapes
    intermediate_dtypes += [[op_inputs_dtype[0]]] * 2 
    #loops
    loops = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR = loops+ '[' + var_intermediate[0] + '=' + 'exp(' + var_inputs + ')-exp(-' + var_inputs + ');];'
    IR+= loops + '[' + var_intermediate[1] + '=' + 'exp(' + var_inputs + ')+exp(-' + var_inputs + ');];'
    IR += loops + '[' + var_outputs + '=' + var_intermediate[0] + '/' + var_intermediate[1] + ';];'
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def log_softmax_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #log_softmax has one input and one output, kwargs: dim=1
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    dim = op_kwargs['dim']
    if dim<0:
        dim += len(op_inputs_shape[0])
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    if dim != 0:
        idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
        loops = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
        for i in range(1, index_len):
            loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    else:
        idx = idx + generate_loop_bind(op_outputs_shape[0][0])
        loops = ''
        for i in range(0, index_len-1):
            loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
        loops += 'B^{' + str(op_outputs_shape[0][-1]) + '}_{tx=0}'
    if loops=='':
        loops='B^{1}_{tx=0}'
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_names, name_start_idx = generate_names(2, name_start_idx)
    temp_intermediate_shapes = [[torch.Size(op_outputs_shape[0][:dim]+op_outputs_shape[0][dim+1:])]]
    var_intermediate=[generate_var_IR(temp_intermediate_names[0], idx[:dim]+idx[dim+1:], op_inputs_dtype[0], temp_intermediate_shapes[0][0]),
                        generate_var_IR(temp_intermediate_names[1], idx[:dim]+idx[dim+1:], op_inputs_dtype[0], temp_intermediate_shapes[0][0])]
    intermediate_names += temp_intermediate_names
    intermediate_shapes += temp_intermediate_shapes*2
    intermediate_dtypes += [[op_inputs_dtype[0]]]*2
    IR = loops + '[' + var_intermediate[0] + '=' + 'max(' + var_intermediate[0] + ',' + var_inputs + ');];'
    IR+=loops + '[' + var_intermediate[1]+ '='+ var_intermediate[1]+'+' + 'exp(' + var_inputs +'-'+var_intermediate[0] + ');];'
    IR+=loops + '[' + var_outputs + '=' + var_inputs +'-'+var_intermediate[0]  + '-log(' + var_intermediate[1] + ');];'
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def gelu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #gelu has one input and one output
    #approximate=none, output=x*0.5*(1+erf(x/sqrt(2)))
    #approximate=tanh, output=x*0.5*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    loops = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    if 'approximate' not in op_kwargs or op_kwargs['approximate'] == 'none':
        #approximate=none, output=x*0.5*(1+erf(x/sqrt(2)))
        IR = loops + '[' + var_outputs + '=' + var_inputs + '*0.5*(1+erf(' + var_inputs + '/sqrt(2)));];'
    elif op_kwargs['approximate'] =='tanh':
        #approximate=tanh, output=x*0.5*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        #tanh(y)=(exp(y)-exp(-y))/(exp(y)+exp(-y))
        temp_intermediate_names, name_start_idx = generate_names(3, name_start_idx)
        temp_intermediate_shapes = [[op_inputs_shape[0]]] * 3
        var_intermediate=[generate_var_IR(temp_intermediate_names[0], idx, op_inputs_dtype[0], temp_intermediate_shapes[0][0]),
                        generate_var_IR(temp_intermediate_names[1], idx, op_inputs_dtype[0], temp_intermediate_shapes[1][0]),
                        generate_var_IR(temp_intermediate_names[2], idx, op_inputs_dtype[0], temp_intermediate_shapes[2][0])]
        intermediate_names += temp_intermediate_names
        intermediate_shapes += temp_intermediate_shapes
        intermediate_dtypes += [[op_inputs_dtype[0]]] * 3 
        IR = loops + '[' + var_intermediate[0] + '=' + 'sqrt(0.6366197466850281)*(' + var_inputs + '+0.044715*' + var_inputs + '**3);];'
        IR += loops + '[' + var_intermediate[1] + '=' + 'exp(' + var_intermediate[0] + ')-exp(-' + var_intermediate[0] + ');];'
        IR += loops + '[' + var_intermediate[2] + '=' + 'exp(' + var_intermediate[0] + ')+exp(-' + var_intermediate[0] + ');];'
        IR += loops + '[' + var_outputs + '=' + var_inputs + '*0.5*(1+' + var_intermediate[1] + '/' + var_intermediate[2] + ');];'
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx    

def selu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #SELU(x)=scale∗(max(0,x)+min(0,alpha∗(exp(x)−1)))
    # with alpha=1.6732632423543772848170429916717 and 
    # scale=1.0507009873554804934193349852946
    #selu has one input and one output
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + '1.0507010221481323*(' + 'max(0,' + var_inputs + ')+min(0,1.6732631921768188*(' + 'exp(' + var_inputs + ')-1)));];'
    return IR

def hardsigmoid_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #hardsigmoid has one input and one output
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + 'max(0,min(1,' + '(' + var_inputs + '+3)/6));];'
    return IR

def softplus_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #softplus has one input and one output
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + 'log(1+exp(' + var_inputs + '));];'
    return IR

def abs_sqrt_log_exp_to_IR(op_name,op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #abs has one input and one output
    index_len = len(op_outputs_shape[0])
    if index_len>0:
        idx = generate_idx_names(index_len-1, 0)
        idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
        var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
        var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
        #loops
        IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
        for i in range(1, index_len):
            IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
        IR += '[' + var_outputs + '=' + op_name +'(' + var_inputs + ');];'
        return IR
    elif index_len==0:
        #if the output shape is empty, then the input is a scalar, just return the input
        var_inputs = generate_var_IR(op_inputs[0], [], op_inputs_dtype[0], op_inputs_shape[0])
        var_outputs = generate_var_IR(op_outputs[0], [], op_outputs_dtype[0], op_outputs_shape[0])
        IR = 'B^{1}_{tx=0}[' + var_outputs + '=' + op_name +'(' + var_inputs + ');];'
        return IR

def elu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs):
    #elu has one input and one output, kwargs: alpha
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    alpha = op_kwargs['alpha']
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + 'if_then_else(' + var_inputs + '<0,' + str(alpha) + '*(' + 'exp(' + var_inputs + ')-1),' + var_inputs + ');];'
    return IR

def hardtanh_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs):
    #hardtanh has one input and one output, kwargs: min, max
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    keys_kwargs = op_kwargs.keys()
    print(keys_kwargs)
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    min_value = op_kwargs['min_val'] if 'min_val' in keys_kwargs else -1.0
    max_value = op_kwargs['max_val'] if 'max_val' in keys_kwargs else 1.0
    IR += '[' + var_outputs + '=' + 'min(max(' + var_inputs + ',' + str(min_value) + '),' + str(max_value) + ');];'
    return IR

def batch_norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #batch_norm has five inputs (input, mean, var, weight, bias) and one output
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    eps=op_kwargs['eps'] if 'eps' in op_kwargs else 1e-5
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = idx + generate_loop_bind(op_outputs_shape[0][-1])
    var_inputs = [generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0]),
                  generate_var_IR(op_inputs[1], idx[1], op_inputs_dtype[1], op_inputs_shape[1]),
                  generate_var_IR(op_inputs[2], idx[1], op_inputs_dtype[2], op_inputs_shape[2]),
                  generate_var_IR(op_inputs[3], idx[1], op_inputs_dtype[3], op_inputs_shape[3]),
                  generate_var_IR(op_inputs[4], idx[1], op_inputs_dtype[4], op_inputs_shape[4])]
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
    var_intermediate = generate_var_IR(temp_intermediate_names[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    intermediate_names += temp_intermediate_names
    intermediate_shapes += [[op_inputs_shape[0]]]
    intermediate_dtypes += [[op_inputs_dtype[0]]]
    #loops
    loops = ''
    for i in range(0, index_len-1):
        loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    loops += 'B^{' + str(op_outputs_shape[0][-1]) + '}_{tx=0}'
    IR = loops + '[' + var_intermediate + '=(' + var_inputs[0] + '-' + var_inputs[1] + ')/sqrt(' + var_inputs[2] + '+'+str(eps)+');];'
    IR += loops + '[' + var_outputs + '=' +var_inputs[3]+'*'+ var_intermediate + '+' + var_inputs[4] + ';];'
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def instance_norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #instance_norm has seven inputs (input, None, None, weight, bias, use_input_stats, momentum, eps) and one output
    #temp intermediate: mean_sum, mean, var_sum, var, \hat{x}
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    eps = op_kwargs['eps'] if 'eps' in op_kwargs else 1e-5
    index_len = len(op_outputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    temp_intermediate_names, name_start_idx = generate_names(5, name_start_idx)
    temp_intermediate_shape = [[op_inputs_shape[0][:2]],
                               [op_inputs_shape[0][:2]],
                               [op_inputs_shape[0][:2]],
                               [op_inputs_shape[0][:2]],
                               [op_inputs_shape[0]]]
    temp_kwargs = {'dim': tuple(x for x in range(len(op_inputs_shape[0])) if x!=1 and x!=0)}
    input_idx, intermediate_idx, first_loops, second_loops, div_num, _,_, _=mean_var_idx_and_loops(temp_kwargs, op_inputs_shape, [op_inputs_shape[0][:2]], idx_prefix)
    var_inputs = [generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0]),
                     generate_var_IR(op_inputs[3], input_idx[1], op_inputs_dtype[3], op_inputs_shape[3]),
                     generate_var_IR(op_inputs[4], input_idx[1], op_inputs_dtype[4], op_inputs_shape[4])]
    var_outputs = generate_var_IR(op_outputs[0], input_idx, op_outputs_dtype[0], op_outputs_shape[0])
    # print(f'op_outputs[0]:{op_outputs[0]},var_outputs:{var_outputs}')
    var_intermediate = [generate_var_IR(temp_intermediate_names[0], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape[0][0]),
                        generate_var_IR(temp_intermediate_names[1], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape[1][0]),
                        generate_var_IR(temp_intermediate_names[2], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape[2][0]),
                        generate_var_IR(temp_intermediate_names[3], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape[3][0]),
                        generate_var_IR(temp_intermediate_names[4], input_idx, op_inputs_dtype[0], temp_intermediate_shape[4][0])]
    #mean calculation
    IR = first_loops + '[' + var_intermediate[0] + '=' +var_intermediate[0]+'+' + var_inputs[0] + ';];'
    IR += second_loops + '[' + var_intermediate[1] + '=' + var_intermediate[0] + '/' + str(div_num) + ';];'
    #variance calculation
    IR += first_loops + '[' + var_intermediate[2] + '=' + var_intermediate[2] + '+(' + var_inputs[0] +'-'+var_intermediate[1] +')**2;];'
    IR += second_loops + '[' + var_intermediate[3] + '=' + var_intermediate[2] + '/' + str(div_num) + ';];'
    #normalization
    IR += first_loops + '[' + var_intermediate[4] + '=' + '(' + var_inputs[0] + '-' + var_intermediate[1] + ')/sqrt(' + var_intermediate[3] + '+' + str(eps) + ');];'
    IR += first_loops + '[' + var_outputs + '=' + var_inputs[1]+ '*' + var_intermediate[4] + '+' + var_inputs[2] + ';];'
    intermediate_names += temp_intermediate_names
    intermediate_shapes += temp_intermediate_shape
    intermediate_dtypes += [[op_inputs_dtype[0]]] * 5

    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def group_norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #group_norm has five inputs (input, num_groups, weight, bias, eps) and one output
    #temp intermediate: mean_sum, mean, var_sum, var, \hat{x}
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    eps = op_kwargs['eps'] if 'eps' in op_kwargs else 1e-5
    num_groups = int(op_inputs[1])
    num_channels_mean_var=op_inputs_shape[0][1] // num_groups
    loop_idx=[op_inputs_shape[0][0], num_groups, num_channels_mean_var] + list(op_inputs_shape[0][2:])
    index_len = len(op_outputs_shape[0])
    idx_prefix = generate_idx_names(index_len, 0)
    temp_intermediate_names, name_start_idx = generate_names(5, name_start_idx)
    mean_var_shape=torch.Size(loop_idx[:2])
    temp_intermediate_shape = [[mean_var_shape],
                               [mean_var_shape],
                               [mean_var_shape],
                               [mean_var_shape],
                               [op_inputs_shape[0]]]
    input_dix=generate_loop_bind(op_inputs_shape[0][0])+[idx_prefix[0]+'*'+str(int(op_inputs_shape[0][1]/num_groups))+'+'+idx_prefix[1]]+ idx_prefix[2:]
    intermediate_dix=generate_loop_bind(op_inputs_shape[0][0])+[idx_prefix[0]]
    div_num = num_channels_mean_var*torch.prod(torch.tensor(op_inputs_shape[0])[list(range(2,index_len))]).item()
    var_inputs = [generate_var_IR(op_inputs[0], input_dix, op_inputs_dtype[0], op_inputs_shape[0]),
                     generate_var_IR(op_inputs[2], input_dix[1], op_inputs_dtype[2], op_inputs_shape[2]),
                     generate_var_IR(op_inputs[3], input_dix[1], op_inputs_dtype[3], op_inputs_shape[3])]
    var_outputs = generate_var_IR(op_outputs[0], input_dix, op_outputs_dtype[0], op_outputs_shape[0])
    var_intermediate = [generate_var_IR(temp_intermediate_names[0], intermediate_dix, op_inputs_dtype[0], temp_intermediate_shape[0][0]),
                        generate_var_IR(temp_intermediate_names[1], intermediate_dix, op_inputs_dtype[0], temp_intermediate_shape[1][0]),
                        generate_var_IR(temp_intermediate_names[2], intermediate_dix, op_inputs_dtype[0], temp_intermediate_shape[2][0]),
                        generate_var_IR(temp_intermediate_names[3], intermediate_dix, op_inputs_dtype[0], temp_intermediate_shape[3][0]),
                        generate_var_IR(temp_intermediate_names[4], input_dix, op_inputs_dtype[0], temp_intermediate_shape[4][0])]
    # loops
    loop_bound = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    first_loops, second_loops = loop_bound, loop_bound
    for i in range(1, index_len+1):
        first_loops += 'L^{' + str(loop_idx[i]) + '}_{' + idx_prefix[i-1] + '=0}'
    second_loops+= 'L^{' + str(loop_idx[1]) + '}_{' + idx_prefix[0] + '=0}'
    # mean calculation
    IR = first_loops + '[' + var_intermediate[0] + '=' + var_intermediate[0] + '+' + var_inputs[0] + ';];'
    IR += second_loops + '[' + var_intermediate[1] + '=' + var_intermediate[0] + '/' + str(div_num) + ';];'
    # variance calculation
    IR += first_loops + '[' + var_intermediate[2] + '=' + var_intermediate[2] + '+(' + var_inputs[0] + '-' + var_intermediate[1] + ')**2;];'    
    IR += second_loops + '[' + var_intermediate[3] + '=' + var_intermediate[2] + '/' + str(div_num) + ';];'
    # normalization
    IR += first_loops + '[' + var_intermediate[4] + '=' + '(' + var_inputs[0] + '-' + var_intermediate[1] + ')/sqrt(' + var_intermediate[3] + '+' + str(eps) + ');];'
    IR += first_loops + '[' + var_outputs + '=' + var_inputs[1] + '*' + var_intermediate[4] + '+' + var_inputs[2] + ';];'
    intermediate_names += temp_intermediate_names
    intermediate_shapes += temp_intermediate_shape
    intermediate_dtypes += [[op_inputs_dtype[0]]] * 5
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx  

def pow_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #pow has two inputs and one output
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + var_inputs + '**' + op_inputs[1] + ';];'
    return IR

def norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #norm has one input and one output, kwargs: p='fro', dim=None, keepdim=False
    #intermediate: square_sum
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    p = op_kwargs['p'] if 'p' in op_kwargs else 'fro'
    index_len = len(op_inputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
    input_idx, output_idx, first_loops, second_loops, _, intermediate_idx,intermediate_idx2, temp_intermediate_shape=mean_var_idx_and_loops(op_kwargs, op_inputs_shape, op_outputs_shape, idx_prefix)
    var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    var_intermediate = generate_var_IR(temp_intermediate_names[0], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape)
    var_intermediate2 = generate_var_IR(temp_intermediate_names[0], intermediate_idx2, op_inputs_dtype[0], temp_intermediate_shape)
    if p=='fro' or p==2:
        IR= first_loops + '[' + var_intermediate2 + '=' + var_intermediate2 + '+' + var_inputs + '**2;];'
        IR += second_loops + '[' + var_outputs + '=sqrt(' + var_intermediate + ');];'
    intermediate_names += temp_intermediate_names
    intermediate_shapes += [[temp_intermediate_shape]]
    intermediate_dtypes += [[op_inputs_dtype[0]]]
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def sum_prod_to_IR(op_name, op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #sum has one input and one output, kwargs: dim=None, keepdim=False
    keepdim = op_kwargs['keepdim'] if 'keepdim' in op_kwargs else False
    index_len = len(op_inputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    input_idx, output_idx, first_loops, second_loops, _, intermediate_idx, intermediate_idx2, temp_intermediate_shape=mean_var_idx_and_loops(op_kwargs, op_inputs_shape, op_outputs_shape, idx_prefix)
    var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    if keepdim:
        #intermediate: sum
        intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
        temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
        temp_intermediate_shapes = [[temp_intermediate_shape]]
        var_intermediate = generate_var_IR(temp_intermediate_names[0], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shapes[0][0])
        var_intermediate2 = generate_var_IR(temp_intermediate_names[0], intermediate_idx2, op_inputs_dtype[0], temp_intermediate_shapes[0][0])
        if op_name == 'sum':
            IR = first_loops + '[' + var_intermediate2 + '=' + var_intermediate2+'+'+var_inputs + ';];'
        elif op_name == 'prod':
            IR = first_loops + '[' + var_intermediate + '=' + var_intermediate+'*'+var_inputs + ';];'
        IR += second_loops + '[' + var_outputs + '=' + var_intermediate + ';];'
        intermediate_names += temp_intermediate_names
        intermediate_shapes += temp_intermediate_shapes
        intermediate_dtypes += [[op_inputs_dtype[0]]]
        return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx
    else:
        #no intermediate
        if op_name =='sum':
            IR = first_loops + '[' + var_outputs + '=' + var_outputs + '+' + var_inputs + ';];'
        elif op_name == 'prod':
            IR = first_loops + '[' + var_outputs + '=' + var_outputs + '*' + var_inputs + ';];'
        return IR, [intermediate_info[0], intermediate_info[1], intermediate_info[2]], name_start_idx

def logsumexp_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #sum has one input and one output, kwargs: dim=None, keepdim=False
    keepdim = op_kwargs['keepdim'] if 'keepdim' in op_kwargs else False
    index_len = len(op_inputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    input_idx, output_idx, first_loops, second_loops, _, intermediate_idx,intermediate_idx2, temp_intermediate_shape=mean_var_idx_and_loops(op_kwargs, op_inputs_shape, op_outputs_shape, idx_prefix)
    var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
    temp_intermediate_shapes = [[temp_intermediate_shape]]
    var_intermediate = generate_var_IR(temp_intermediate_names[0], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shapes[0][0])
    var_intermediate2 = generate_var_IR(temp_intermediate_names[0], intermediate_idx2, op_inputs_dtype[0], temp_intermediate_shapes[0][0])
    IR = first_loops + '[' + var_intermediate2 + '=' + var_intermediate2+'+exp('+var_inputs + ');];'
    IR += second_loops + '[' + var_outputs + '=log(' + var_intermediate + ');];'
    intermediate_names += temp_intermediate_names
    intermediate_shapes += temp_intermediate_shapes
    intermediate_dtypes += [[op_inputs_dtype[0]]]
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def layer_norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #layer_norm has five inputs: input, normalized_shape=[*, input shape], weight, bias, eps and one output
    #intermediate: mean_sum, mean, var_sum, var, \hat{x}
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    eps=op_kwargs['eps'] if 'eps' in op_kwargs else 1e-5
    normalized_shape=eval(op_inputs[1])
    index_len = len(op_outputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    temp_intermediate_names, name_start_idx = generate_names(5, name_start_idx)
    temp_intermediate_shape = [[op_inputs_shape[0][:index_len-len(normalized_shape)]],
                               [op_inputs_shape[0][:index_len-len(normalized_shape)]],
                               [op_inputs_shape[0][:index_len-len(normalized_shape)]],
                               [op_inputs_shape[0][:index_len-len(normalized_shape)]],
                               [op_inputs_shape[0]]]
    temp_kwargs = {'dim': tuple(x for x in range(index_len-len(normalized_shape), index_len))}
    input_idx, intermediate_idx, first_loops, second_loops, div_num, _,_, _=mean_var_idx_and_loops(temp_kwargs, op_inputs_shape, [op_inputs_shape[0][:2]], idx_prefix)
    var_inputs = [generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0]),
                     generate_var_IR(op_inputs[2], input_idx[-len(normalized_shape):], op_inputs_dtype[2], op_inputs_shape[2]),
                     generate_var_IR(op_inputs[3], input_idx[-len(normalized_shape):], op_inputs_dtype[3], op_inputs_shape[3])]
    var_outputs = generate_var_IR(op_outputs[0], input_idx, op_outputs_dtype[0], op_outputs_shape[0])
    var_intermediate = [generate_var_IR(temp_intermediate_names[0], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape[0][0]),
                        generate_var_IR(temp_intermediate_names[1], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape[1][0]),
                        generate_var_IR(temp_intermediate_names[2], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape[2][0]),
                        generate_var_IR(temp_intermediate_names[3], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shape[3][0]),
                        generate_var_IR(temp_intermediate_names[4], input_idx, op_inputs_dtype[0], temp_intermediate_shape[4][0])]
    #mean calculation
    IR = first_loops + '[' + var_intermediate[0] + '=' +var_intermediate[0]+'+' + var_inputs[0] + ';];'
    IR += second_loops + '[' + var_intermediate[1] + '=' + var_intermediate[0] + '/' + str(div_num) + ';];'
    #variance calculation
    IR += first_loops + '[' + var_intermediate[2] + '=' + var_intermediate[2] + '+(' + var_inputs[0] +'-'+var_intermediate[1] +')**2;];'
    IR += second_loops + '[' + var_intermediate[3] + '=' + var_intermediate[2] + '/' + str(div_num) + ';];'
    #normalization
    IR += first_loops + '[' + var_intermediate[4] + '=' + '(' + var_inputs[0] + '-' + var_intermediate[1] + ')/sqrt(' + var_intermediate[3] + '+' + str(eps) + ');];'
    IR += first_loops + '[' + var_outputs + '=' + var_inputs[1]+ '*' + var_intermediate[4] + '+' + var_inputs[2] + ';];'
    intermediate_names += temp_intermediate_names
    intermediate_shapes += temp_intermediate_shape
    intermediate_dtypes += [[op_inputs_dtype[0]]] * 5
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def boolean_dispatch_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #boolean_dispatch is maxpooling
    #it has two inputs (input, kernel_size) and one output
    #kwargs: stride=1, padding=0, dilation=1, ceil_mode=False, count_include_pad=False
    #intermediate:padding
    padding = op_kwargs['padding'] if 'padding' in op_kwargs else 0
    stride = op_kwargs['stride'] if 'stride' in op_kwargs else 1
    dilation = op_kwargs['dilation'] if 'dilation' in op_kwargs else 1
    kernel_size = op_inputs[1]
    index_len = len(op_inputs_shape[0])
    second_index_len = (len(op_outputs_shape[0])-2)*2+1
    idx = generate_idx_names(second_index_len, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    final_input_idx=idx[:2]+[idx[i]+'*'+str(stride)+'+'+idx[i-index_len]+'*'+str(dilation) for i in range(2, index_len)]
    # print(f'final_input_idx:{final_input_idx}, idx:{idx}')
    second_loops = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        second_loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    if isinstance(eval(kernel_size), int):
        for i in range(index_len, second_index_len+1):
            second_loops += 'L^{' + str(eval(kernel_size)) + '}_{' + idx[i] + '=0}'
    else:
        for i in range(index_len, second_index_len+1):
            second_loops += 'L^{' + str(eval(kernel_size)[0]) + '}_{' + idx[i] + '=0}'
    var_outputs = generate_var_IR(op_outputs[0], idx[:index_len], op_outputs_dtype[0], op_outputs_shape[0])
    if padding>0:
        intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
        var_inputs = generate_var_IR(op_inputs[0], idx[:2]+[idx[i]+'-'+str(padding) for i in range(2,index_len)], op_inputs_dtype[0], op_inputs_shape[0])
        temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
        temp_intermediate_shape = op_inputs_shape[0][:2]+torch.Size([op_inputs_shape[0][i]+2*padding for i in range(2, len(op_inputs_shape[0]))])
        var_intermediates = [generate_var_IR(temp_intermediate_names[0], idx[:index_len], op_inputs_dtype[0], temp_intermediate_shape),
                            generate_var_IR(temp_intermediate_names[0], final_input_idx, op_inputs_dtype[0], op_inputs_shape[0])]
        first_loops = 'B^{' + str(temp_intermediate_shape[0]) + '}_{tx=0}'
        for i in range(1, index_len):
            first_loops += 'L^{' + str(temp_intermediate_shape[i]) + '}_{' + idx[i] + '=0}'
        IR= first_loops + '[' + var_intermediates[0] + '=if_then_else('
        for i in range(2, index_len-1):
            IR += str(padding)+'<=' + idx[i] + '<' + str(padding+op_inputs_shape[0][i]) + '&'
        IR += str(padding)+'<=' + idx[index_len-1] + '<' + str(padding+op_inputs_shape[0][-1]) + ', ' + var_inputs + ', -inf);];'
        IR += second_loops + '[' + var_outputs + '=' + 'max(' + var_outputs + ', ' + var_intermediates[1] + ');];'
        intermediate_names += temp_intermediate_names
        intermediate_shapes += [[temp_intermediate_shape]]
        intermediate_dtypes += [[op_inputs_dtype[0]]]
        return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx
    elif padding==0:
        var_inputs = generate_var_IR(op_inputs[0], final_input_idx, op_inputs_dtype[0], op_inputs_shape[0])
        IR=second_loops + '[' + var_outputs + '=' + 'max(' + var_outputs + ', ' + var_inputs + ');];'
        return IR, [intermediate_info[0], intermediate_info[1], intermediate_info[2]], name_start_idx

def avg_pool_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #avg_pool has six inputs: input, kernel_size, stride, padding, ceil_mode, count_include_pad and one output
    #intermediate:padding, pool_sum
    kernel_size = eval(op_inputs[1])[0] if isinstance(eval(op_inputs[1]), tuple) else eval(op_inputs[1])
    stride = eval(op_inputs[2])[0] if isinstance(eval(op_inputs[2]), tuple) else eval(op_inputs[2])
    padding = eval(op_inputs[3])[0] if isinstance(eval(op_inputs[3]), tuple) else eval(op_inputs[3])
    count_include_pad = eval(op_inputs[5]) if isinstance(eval(op_inputs[5]), bool) else True
    # print('kernel_size:', kernel_size, 'stride:', stride, 'padding:', padding, 'count_include_pad:', count_include_pad)
    index_len = len(op_inputs_shape[0])
    second_index_len = (len(op_outputs_shape[0])-2)*2+1
    idx = generate_idx_names(second_index_len, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0])+ idx
    second_loops = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    second_loop_final=''
    for i in range(1, index_len):
        second_loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    for i in range(index_len, second_index_len+1):
        second_loop_final += 'L^{' + str(kernel_size) + '}_{' + idx[i] + '=0}'
    var_outputs = generate_var_IR(op_outputs[0], idx[:index_len], op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_name_sum, name_start_idx = generate_names(1, name_start_idx)
    var_intermediate_sum = generate_var_IR(temp_intermediate_name_sum[0], idx[:index_len], op_outputs_dtype[0], op_outputs_shape[0])
    final_input_idx=idx[:2]+[idx[i]+'*'+str(stride)+'+'+idx[i-index_len] for i in range(2, index_len)]
    if padding>0:
        intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
        input_idx= idx[:2]+[idx[i]+'-'+str(padding) for i in range(2,index_len)]
        var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
        temp_intermediate_names_padding, name_start_idx = generate_names(1, name_start_idx)
        temp_intermediate_shape_padding = op_inputs_shape[0][:2]+torch.Size([op_inputs_shape[0][i]+2*padding for i in range(2, len(op_inputs_shape[0]))])
        var_intermediates_padding = [generate_var_IR(temp_intermediate_names_padding[0], idx[:index_len], op_inputs_dtype[0], temp_intermediate_shape_padding),
                            generate_var_IR(temp_intermediate_names_padding[0], final_input_idx, op_outputs_dtype[0], temp_intermediate_shape_padding)]
        first_loops = 'B^{' + str(temp_intermediate_shape_padding[0]) + '}_{tx=0}'
        for i in range(1, index_len):
            first_loops += 'L^{' + str(temp_intermediate_shape_padding[i]) + '}_{' + idx[i] + '=0}'
        IR= first_loops + '[' + var_intermediates_padding[0] + '=if_then_else('
        for i in range(2, index_len-1):
            IR += str(padding)+'<=' + idx[i] + '<' + str(padding+op_inputs_shape[0][i]) + '&'
        IR += str(padding)+'<=' + idx[index_len-1] + '<' + str(padding+op_inputs_shape[0][-1]) + ', ' + var_inputs + ', 0);];'
        IR += second_loops + second_loop_final + '[' + var_intermediate_sum + '=' + var_intermediate_sum + '+' + var_intermediates_padding[1] + ';];'
        IR += second_loops  + '[' + var_outputs + '=' + var_intermediate_sum + '/'
        if count_include_pad:
            if index_len-1>2:
                IR+='('
                for i in range(2, index_len-1):
                    IR += '(min('+ str(kernel_size-padding-1)+','+ str(op_inputs_shape[0][i]+padding-1) +'-'+idx[i]+'*'+str(stride) +')+'+str(padding+1)+')*'
                IR += '(min('+ str(kernel_size-padding-1)+','+ str(op_inputs_shape[0][index_len-1]+padding-1)+'-'+idx[index_len-1] +'*'+str(stride) +')+'+str(padding+1)+'));];'
            else:
                IR += '(min('+ str(kernel_size-padding-1)+','+ str(op_inputs_shape[0][index_len-1]+padding-1)+'-'+idx[index_len-1] +'*'+str(stride) +')+'+str(padding+1)+');];'
        else:
            IR+='(max('
            for i in range(2, index_len-1):
                IR += '(min('+idx[i]+'*'+str(stride)+'+'+str(kernel_size-padding-1)+','+ str(op_inputs_shape[0][i]-1)+')+'+str(padding+1)+'-max('+str(padding)+'-'+idx[i]+'*'+str(stride)+',0)-'+idx[i]+'*'+str(stride)+')*'
            IR += '(min('+idx[index_len-1]+'*'+str(stride)+'+'+str(kernel_size-padding-1)+','+ str(op_inputs_shape[0][index_len-1]-1)+')+'+str(padding+1)+'-max('+str(padding)+'-'+idx[index_len-1]+'*'+str(stride)+',0)-'+idx[index_len-1]+'*'+str(stride)+'),1);];'
        intermediate_names += temp_intermediate_names_padding + temp_intermediate_name_sum
        intermediate_shapes += [[op_outputs_shape[0]], [temp_intermediate_shape_padding], [temp_intermediate_shape_padding]]
        intermediate_dtypes += [[op_inputs_dtype[0]]] * 2
        return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx
    elif padding==0:
        var_inputs = generate_var_IR(op_inputs[0], final_input_idx, op_inputs_dtype[0], op_inputs_shape[0])
        IR=second_loops + second_loop_final + '[' + var_intermediate_sum + '=' + var_intermediate_sum + '+' + var_inputs + ';];'
        IR += second_loops  + '[' + var_outputs + '=' + var_intermediate_sum + '/'
        if count_include_pad:
            if index_len-1>2:
                IR+='('
                for i in range(2, index_len-1):
                    IR += '(min('+ str(kernel_size-padding-1)+','+ str(op_inputs_shape[0][i]+padding-1) +'-'+idx[i]+'*'+str(stride) +')+'+str(padding+1)+')*'
                IR += '(min('+ str(kernel_size-padding-1)+','+ str(op_inputs_shape[0][index_len-1]+padding-1)+'-'+idx[index_len-1] +'*'+str(stride) +')+'+str(padding+1)+'));];'
            else:
                IR += '(min('+ str(kernel_size-padding-1)+','+ str(op_inputs_shape[0][index_len-1]+padding-1)+'-'+idx[index_len-1] +'*'+str(stride) +')+'+str(padding+1)+');];'
        else:
            IR+='(max('
            for i in range(2, index_len-1):
                IR += '(min('+idx[i]+',0)*'+str(stride)+'+min('+idx[i]+'*'+str(stride)+'+'+str(kernel_size-1)+','+ str(op_inputs_shape[0][i]-1)+')+1-'+idx[i]+'*'+str(stride)+')*'
            IR += '(min('+idx[index_len-1]+',0)*'+str(stride)+'+min('+idx[index_len-1]+'*'+str(stride)+'+'+str(kernel_size-1)+','+ str(op_inputs_shape[0][index_len-1]-1)+')+1-'+idx[index_len-1]+'*'+str(stride)+'),1);];'
        return IR, [intermediate_info[0], intermediate_info[1], intermediate_info[2]], name_start_idx

# def max_min_to_IR(op_name, op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
#     #max has one input and one output, kwargs: dim=None, keepdim=False
#     # one output has two values: max and index
#     # we just focus on the final output is max or min value
#     if len(op_inputs)==1:
#         keepdim = op_kwargs['keepdim'] if 'keepdim' in op_kwargs else False
#         dim = op_kwargs['dim']
#         if dim<0:
#             dim += len(op_inputs_shape[0])
#         index_len = len(op_inputs_shape[0])
#         idx_prefix = generate_idx_names(index_len-1, 0)
#         intermediate_index_name, name_start_idx = generate_names(1, name_start_idx)
#         input_idx, output_idx, first_loops, second_loops, _, intermediate_idx, temp_intermediate_shape=mean_var_idx_and_loops(op_kwargs, op_inputs_shape, op_outputs_shape[0], idx_prefix)
#         var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
#         var_intermediate_index = generate_var_IR(intermediate_index_name[0], output_idx, str(op_outputs_dtype[0][1]), op_outputs_shape[0][1])
#         var_outputs = generate_var_IR(op_outputs[0], output_idx, str(op_outputs_dtype[0][0]), op_outputs_shape[0][0])
#         if keepdim:
#             #intermediate: max
#             intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
#             temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
#             temp_intermediate_shapes = [[temp_intermediate_shape]]
#             var_intermediate = generate_var_IR(temp_intermediate_names[0], intermediate_idx, str(op_outputs_dtype[0][0]), temp_intermediate_shapes[0][0])
#             if op_name == 'max':
#                 IR = first_loops + '[' + var_intermediate + '=max(' + var_intermediate+','+var_inputs + ');'+var_intermediate_index+'=0;];'
#                 IR += second_loops + '[' + var_outputs + '=' + var_intermediate + ';];'
#                 # IR += first_loops + '[' + var_intermediate_index + '=if_then_else(' +var_inputs+'>='+var_intermediate +','+ input_idx[dim]+','+var_intermediate_index +');];'
#             elif op_name == 'min':
#                 IR = first_loops + '[' + var_intermediate + '=min(' + var_intermediate+','+var_inputs + ');'+var_intermediate_index+'=0;];'
#                 IR += second_loops + '[' + var_outputs + '=' + var_intermediate + ';];'
#                 # IR += first_loops + '[' + var_intermediate_index + '=if_then_else(' +var_inputs+'<='+var_intermediate +','+ input_idx[dim]+','+var_intermediate_index +');];'
#             intermediate_names += temp_intermediate_names
#             intermediate_shapes += temp_intermediate_shapes
#             intermediate_dtypes += [[op_inputs_dtype[0]]]
#             return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx
#         else:
#             #no intermediate
#             if op_name == 'max':
#                 IR = first_loops + '[' + var_outputs + '=max(' + var_outputs + ',' + var_inputs + ');'+var_intermediate_index+'=0;];'
#                 # IR+= first_loops + '[' + var_intermediate_index+ '=if_then_else(' +var_inputs+'>='+var_outputs +','+ input_idx[dim]+','+var_intermediate_index +');];'
#             elif op_name == 'min':
#                 IR = first_loops + '[' + var_outputs + '=min(' + var_outputs + ',' + var_inputs + ');'+var_intermediate_index+'=0;];'
#                 # IR+= first_loops + '[' + var_intermediate_index + '=if_then_else(' +var_inputs+'<='+var_outputs +','+ input_idx[dim]+','+var_intermediate_index +');];'
#             return IR, [intermediate_info[0], intermediate_info[1], intermediate_info[2]], name_start_idx
#     elif len(op_inputs)==2:
#         #min/max has two inputs and one output
#         index_len = len(op_outputs_shape[0])
#         idx = generate_idx_names(index_len-1, 0)
#         idx = generate_loop_bind(op_outputs_shape[0][0]) + idx 
#         input_idx = generate_input_idx_for_mul_add_sub_truediv(idx, op_inputs_shape, op_outputs_shape)
#         var_inputs = [generate_var_IR(op_inputs[0], input_idx[0], op_inputs_dtype[0], op_inputs_shape[0]),
#                     generate_var_IR(op_inputs[1], input_idx[1], op_inputs_dtype[1], op_inputs_shape[1])]
#         var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_dtype[0])
#         #loops
#         IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
#         for i in range(1, index_len):
#             IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
#         if op_name == 'max':
#             IR += '[' + var_outputs + '=' + 'max(' + var_inputs[0] + ',' + var_inputs[1] + ');];'
#         elif op_name == 'min':
#             IR += '[' + var_outputs + '=' + 'min(' + var_inputs[0] + ',' + var_inputs[1] + ');];'
#         return IR, [intermediate_info[0], intermediate_info[1], intermediate_info[2]], name_start_idx

def max_min_to_IR(op_name, op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #max has one input and one output, kwargs: dim=None, keepdim=False
    # one output has two values: max and index
    # we just focus on the final output is max or min value
    if len(op_inputs)==1:
        keepdim = op_kwargs['keepdim'] if 'keepdim' in op_kwargs else False
        dim = op_kwargs['dim']
        if dim<0:
            dim += len(op_inputs_shape[0])
        index_len = len(op_inputs_shape[0])
        idx_prefix = generate_idx_names(index_len-1, 0)
        intermediate_index_name, name_start_idx = generate_names(1, name_start_idx)
        input_idx, output_idx, first_loops, second_loops, _, intermediate_idx,intermediate_idx2, temp_intermediate_shape=mean_var_idx_and_loops(op_kwargs, op_inputs_shape, op_outputs_shape, idx_prefix)
        print(f'intermediate_idx:{intermediate_idx},intermediate_idx2:{intermediate_idx2}')
        var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
        var_outputs = generate_var_IR(op_outputs[0], output_idx, str(op_outputs_dtype[0]), op_outputs_shape[0])
        if keepdim:
            #intermediate: max
            intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
            temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
            temp_intermediate_shapes = [[temp_intermediate_shape]]
            var_intermediate = generate_var_IR(temp_intermediate_names[0], intermediate_idx, str(op_outputs_dtype[0]), temp_intermediate_shapes[0][0])
            var_intermediate2 = generate_var_IR(temp_intermediate_names[0], intermediate_idx2, str(op_outputs_dtype[0]), temp_intermediate_shapes[0][0])
            if op_name == 'max':
                IR = first_loops + '[' + var_intermediate2 + '=max(' + var_intermediate2+','+var_inputs + ');];'
                IR += second_loops + '[' + var_outputs + '=' + var_intermediate + ';];'
                # IR += first_loops + '[' + var_intermediate_index + '=if_then_else(' +var_inputs+'>='+var_intermediate +','+ input_idx[dim]+','+var_intermediate_index +');];'
            elif op_name == 'min':
                IR = first_loops + '[' + var_intermediate2 + '=min(' + var_intermediate2+','+var_inputs + ');];'
                IR += second_loops + '[' + var_outputs + '=' + var_intermediate + ';];'
                # IR += first_loops + '[' + var_intermediate_index + '=if_then_else(' +var_inputs+'<='+var_intermediate +','+ input_idx[dim]+','+var_intermediate_index +');];'
            intermediate_names += temp_intermediate_names
            intermediate_shapes += temp_intermediate_shapes
            intermediate_dtypes += [[op_inputs_dtype[0]]]
            return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx
        else:
            #no intermediate
            if op_name == 'max':
                IR = first_loops + '[' + var_outputs + '=max(' + var_outputs + ',' + var_inputs + ');];'
                # IR+= first_loops + '[' + var_intermediate_index+ '=if_then_else(' +var_inputs+'>='+var_outputs +','+ input_idx[dim]+','+var_intermediate_index +');];'
            elif op_name == 'min':
                IR = first_loops + '[' + var_outputs + '=min(' + var_outputs + ',' + var_inputs + ');];'
                # IR+= first_loops + '[' + var_intermediate_index + '=if_then_else(' +var_inputs+'<='+var_outputs +','+ input_idx[dim]+','+var_intermediate_index +');];'
            return IR, [intermediate_info[0], intermediate_info[1], intermediate_info[2]], name_start_idx
    elif len(op_inputs)==2:
        #min/max has two inputs and one output
        index_len = len(op_outputs_shape[0])
        idx = generate_idx_names(index_len-1, 0)
        idx = generate_loop_bind(op_outputs_shape[0][0]) + idx 
        input_idx = generate_input_idx_for_mul_add_sub_truediv(idx, op_inputs_shape, op_outputs_shape)
        var_inputs = [generate_var_IR(op_inputs[0], input_idx[0], op_inputs_dtype[0], op_inputs_shape[0]),
                    generate_var_IR(op_inputs[1], input_idx[1], op_inputs_dtype[1], op_inputs_shape[1])]
        var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_dtype[0])
        #loops
        IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
        for i in range(1, index_len):
            IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
        if op_name == 'max':
            IR += '[' + var_outputs + '=' + 'max(' + var_inputs[0] + ',' + var_inputs[1] + ');];'
        elif op_name == 'min':
            IR += '[' + var_outputs + '=' + 'min(' + var_inputs[0] + ',' + var_inputs[1] + ');];'
        return IR, [intermediate_info[0], intermediate_info[1], intermediate_info[2]], name_start_idx

def argmax_argmin_to_IR(op_name,op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #argmax has one input and one output, kwargs: dim=None, keepdim=False
    dim = op_kwargs['dim']
    if dim<0:
        dim += len(op_inputs_shape[0])
    index_len = len(op_inputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    input_idx, output_idx, first_loops, _, _, intermediate_idx,intermediate_idx2, temp_intermediate_shape=mean_var_idx_and_loops(op_kwargs, op_inputs_shape, op_outputs_shape, idx_prefix)
    var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])                
    #intermediate: max
    intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
    temp_intermediate_names, name_start_idx = generate_names(2, name_start_idx)
    temp_intermediate_shapes = [[temp_intermediate_shape]]
    var_intermediate = generate_var_IR(temp_intermediate_names[0], intermediate_idx, op_inputs_dtype[0], temp_intermediate_shapes[0][0])
    var_intermediate2 = generate_var_IR(temp_intermediate_names[0], intermediate_idx2, op_inputs_dtype[0], temp_intermediate_shapes[0][0])
    if op_name == 'argmax':
        IR = first_loops + '[' + var_intermediate2 + '=max(' + var_intermediate2+','+var_inputs + ');'+var_outputs+'=0;];'
        IR += first_loops + '[' + var_outputs + '=if_then_else(' +var_inputs+'>='+var_intermediate2 +','+ input_idx[dim]+','+var_outputs +');];'
    elif op_name == 'argmin':
        IR = first_loops + '[' + var_intermediate2 + '=min(' + var_intermediate2+','+var_inputs + ');'+var_outputs+'=0;];'
        IR += first_loops + '[' + var_outputs + '=if_then_else(' +var_inputs+'<='+var_intermediate2 +','+ input_idx[dim]+','+var_outputs +');];'
    intermediate_names += temp_intermediate_names
    intermediate_shapes += temp_intermediate_shapes
    intermediate_dtypes += [[op_inputs_dtype[0]]]
    return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx

def getitem_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #getitem has two inputs (input, index) and one output
    #or getitem has two inputs (input, slice(start, stop, step)) and one output
    index_len = len(op_outputs_shape[0])
    if len(op_inputs)==2:
        index_info= eval(op_inputs[1])
        idx = generate_idx_names(index_len-1, 0)
        idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
        var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
        if isinstance(index_info, int):
            var_inputs = generate_var_IR(op_inputs[0], [str(index_info)]+idx, str(op_inputs_dtype[0][index_info]), torch.Size([1])+op_inputs_shape[0][0])
            #loops
            IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
            for i in range(1, index_len):
                IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
            IR += '[' + var_outputs + '=' + var_inputs + ';];'
            return IR
        elif isinstance(index_info, slice):
            start = index_info.start if index_info.start is not None else 0
            stop = index_info.stop if index_info.stop is not None else op_inputs_shape[0][0]
            step = index_info.step if index_info.step is not None else 1
            #loops, and don't consider step
            if step == 1:
                if start==0:
                    input_idx=idx
                else:
                    input_idx = [idx[0]+'+'+str(start)]+idx[1:]
                var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
                IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
                for i in range(1, index_len):
                    IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
                IR += '[' + var_outputs + '=' + var_inputs + ';];'
                return IR
    else:
        return ''

def conv_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #conv has five inputs: input, None(?), stride, padding, dilation, groups
    #one output
    #intermediate: padding
    stride = eval(op_inputs[3])
    padding = eval(op_inputs[4])
    dilation = eval(op_inputs[5])
    groups = eval(op_inputs[6])
    index_len = len(op_inputs_shape[0])+len(op_inputs_shape[1])-1
    is_Padding, padding_idx=is_num_in_tuple(0, padding)
    is_stride, stride_idx=is_num_in_tuple(1, stride)
    is_dilation, dilation_idx=is_num_in_tuple(1, dilation)
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    second_loops = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, len(op_inputs_shape[0])):
        second_loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    for i in range(0, len(op_inputs_shape[1])-1):
        second_loops += 'L^{' + str(op_inputs_shape[1][i+1]) + '}_{' + idx[len(op_inputs_shape[0])+i] + '=0}'
    var_outputs = generate_var_IR(op_outputs[0], idx[:len(op_outputs_shape[0])], op_outputs_dtype[0], op_outputs_shape[0])
    input_idx_2=[idx[1]]+idx[len(op_inputs_shape[0]):]
    # print(f'input_idx_2:{input_idx_2}')
    if not is_stride and not is_dilation:
        if groups==1:
            final_input_idx = [idx[0]]+[idx[len(op_outputs_shape[0])]]+[idx[i]+'+'+idx[i+len(op_inputs_shape[1])-1] for i in range(2, len(op_outputs_shape[0]))]
        elif groups>1: 
            final_input_idx = [idx[0]]+[idx[1]]+[idx[i]+'+'+idx[i+len(op_inputs_shape[1])-1] for i in range(2, len(op_outputs_shape[0]))]
    else:
        if groups==1:
            final_input_idx = [idx[0]]+[idx[len(op_outputs_shape[0])]]
        elif groups>1:
            final_input_idx = [idx[0]]+[idx[1]]
        for i in range(2, len(op_outputs_shape[0])):
            temp_input_idx=[idx[i],idx[i+len(op_inputs_shape[1])-1]]
            if is_stride and i-2 in stride_idx:
                temp_input_idx[0] += '*'+str(stride[i-2])
            if is_dilation and i-2 in dilation_idx:
                temp_input_idx[1] += '*'+str(dilation[i-2])
            final_input_idx.append(temp_input_idx[0]+'+'+temp_input_idx[1])    
    if is_Padding:
        intermediate_names, intermediate_shapes, intermediate_dtypes = intermediate_info
        input_idx_1=idx[:2]
        for i in range(len(padding)):
            if i in padding_idx:
                input_idx_1 += [idx[i+2]+'-'+str(padding[i])]
            else:
                input_idx_1 += [idx[i+2]]     
        var_inputs = [generate_var_IR(op_inputs[0], input_idx_1, op_inputs_dtype[0], op_inputs_shape[0]),
                        generate_var_IR(op_inputs[1], input_idx_2, op_inputs_dtype[1], op_inputs_shape[1])]
        temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
        temp_intermediate_shape = op_inputs_shape[0][:2]
        temp_intermediate_shape2=[]
        for i in range(0, len(padding)):
            if i in padding_idx:
                temp_intermediate_shape2.append(op_inputs_shape[0][i+2]+2*padding[i])
            else:
                temp_intermediate_shape2.append(op_inputs_shape[0][i+2])
        temp_intermediate_shape += torch.Size(temp_intermediate_shape2)
        var_intermediates = [generate_var_IR(temp_intermediate_names[0], idx[:len(op_inputs_shape[0])], op_inputs_dtype[0], temp_intermediate_shape),
                            generate_var_IR(temp_intermediate_names[0], final_input_idx, op_inputs_dtype[0], op_inputs_shape[0])]
        first_loops = 'B^{' + str(temp_intermediate_shape[0]) + '}_{tx=0}'
        for i in range(1, len(op_inputs_shape[0])):
            first_loops += 'L^{' + str(temp_intermediate_shape[i]) + '}_{' + idx[i] + '=0}'
        IR = first_loops + '[' + var_intermediates[0] + '=if_then_else('
        for i in range(len(padding_idx)-1):
            item= padding_idx[i]    
            IR += str(padding[item])+'<=' + idx[item+2] + '<' + str(op_inputs_shape[0][item+2]+padding[item]) + '&'
        IR += str(padding[padding_idx[-1]])+'<=' + idx[padding_idx[-1]+2] + '<' + str(op_inputs_shape[0][padding_idx[-1]+2]+padding[padding_idx[-1]]) + ', ' + var_inputs[0] + ', 0);];'
        IR += second_loops + '[' + var_outputs + '=' + var_outputs + '+' + var_intermediates[1]+ '*'+ var_inputs[1] + ';];'
        intermediate_names += temp_intermediate_names
        intermediate_shapes += [[temp_intermediate_shape]]
        intermediate_dtypes += [[op_inputs_dtype[0]]]
        return IR, [intermediate_names, intermediate_shapes, intermediate_dtypes], name_start_idx
    else:
        var_inputs = [generate_var_IR(op_inputs[0], final_input_idx, op_inputs_dtype[0], op_inputs_shape[0]),
                        generate_var_IR(op_inputs[1], input_idx_2, op_inputs_dtype[1], op_inputs_shape[1])]
        IR = second_loops + '[' + var_outputs + '=' + var_outputs + '+' + var_inputs[1] + '*'+ var_inputs[0] + ';];'
        return IR, [intermediate_info[0], intermediate_info[1], intermediate_info[2]], name_start_idx

def cumsum_to_IR(op_name, op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs):
    #cumsum has one input and one output, kwargs: dim=0, dtype=None
    dim= op_kwargs['dim'] if 'dim' in op_kwargs else 0
    if dim<0:
        dim += len(op_inputs_shape[0])
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0])+ idx
    output_idx=idx[:-1]
    input_idx=idx[:dim]+[idx[-1]]+idx[dim+1:-1]
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, dim+1):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += 'L^{' + str(op_outputs_shape[0][dim]) + '}_{' + idx[dim+1] + '=0}'
    for i in range(dim+2, index_len+1):
        IR += 'L^{' + str(op_outputs_shape[0][i-1]) + '}_{' + idx[i] + '=0}'
    if op_name == 'cumsum':
        IR += '[' + var_outputs + '=' + var_outputs + '+if_then_else('+input_idx[dim]+'<='+output_idx[dim]+',' + var_inputs + ',0);];'
    elif op_name == 'cumprod':
        IR += '[' + var_outputs + '=' + var_outputs + '*if_then_else('+input_idx[dim]+'<='+output_idx[dim]+',' + var_inputs + ',1);];'
    return IR

def flip_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #flip has two inputs (input, dims) and one output
    #dims is a list of integers or an integer
    dims = eval(op_inputs[1]) if isinstance(eval(op_inputs[1]), list) else [eval(op_inputs[1])]
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0])+ idx
    input_idx=[]
    for i in range(0, index_len):
        if i in dims:
            if 'bx' in idx[i]:
                input_idx.append(str(op_inputs_shape[0][i]-1)+'-('+idx[i]+')')
            else:
                input_idx.append(str(op_inputs_shape[0][i]-1)+'-'+idx[i])
        else:
            input_idx.append(idx[i])
    var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + var_inputs + ';];'
    return IR

def select_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #select has three inputs (input, dim, index) and one output
    #dim is an integer, index is an integer
    dim = int(op_inputs[1]) 
    if dim<0:
        dim += len(op_inputs_shape[0])
    index = int(op_inputs[2])
    index_len = len(op_inputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    input_idx, output_idx, _, second_loops, _, _, _,_ = mean_var_idx_and_loops({'dim':dim}, op_inputs_shape, op_outputs_shape, idx_prefix)
    input_idx[dim]=str(index)
    var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    IR=second_loops + '[' + var_outputs + '=' + var_inputs + ';];'
    return IR

def unsqueeze_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #unsqueeze has two inputs (input, dim) and one output
    #dim is an integer
    dim = int(op_inputs[1])
    if dim<0:
        dim += len(op_inputs_shape[0])+1
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx[:dim] + ['0'] + idx[dim:], op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len+1):
        if i<dim:
            IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
        elif i>dim:
            IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i-1] + '=0}'
    IR += '[' + var_outputs + '=' + var_inputs + ';];'
    return IR

def squeeze_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #squeeze has two inputs (input, dim) and one output
    dim = int(op_inputs[1])
    if dim<0:
        dim += len(op_inputs_shape[0])
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_outputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx[:dim]+['0']+idx[dim:], op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=' + var_inputs + ';];'
    return IR

def zeros_like_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #zeros_like has one input and one output
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    IR = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        IR += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR += '[' + var_outputs + '=0;];'
    return IR

def cat_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs):
    #cat has two inputs (input1, input2) and one output
    dim = op_kwargs['dim'] if 'dim' in op_kwargs else 0
    if dim<0:
        dim += len(op_inputs_shape[0])
    index_len = len(op_inputs_shape[0])
    idx_prefix = generate_idx_names(index_len-1, 0)
    idx = [idx_prefix[:dim]+generate_loop_bind(op_inputs_shape[0][dim])+idx_prefix[dim:],
           idx_prefix[:dim]+generate_loop_bind(op_inputs_shape[1][dim])+idx_prefix[dim:],
           idx_prefix[:dim]+[generate_loop_bind(op_inputs_shape[1][dim])[0]+ '+'+str(op_inputs_shape[0][dim])]+idx_prefix[dim:]]
    var_inputs=[generate_var_IR(op_inputs[0], idx[0], op_inputs_dtype[0], op_inputs_shape[0]),
                    generate_var_IR(op_inputs[1], idx[1], op_inputs_dtype[1], op_inputs_shape[1])]
    var_outputs = [generate_var_IR(op_outputs[0], idx[0], op_outputs_dtype[0], op_outputs_shape[0]),
                    generate_var_IR(op_outputs[0], idx[2], op_outputs_dtype[0], op_outputs_shape[0])]
    #loops
    first_loops, second_loops = '', ''
    for i in range(0,dim):
        first_loops+= 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[0][i] + '=0}'
        second_loops+= 'L^{' + str(op_inputs_shape[1][i]) + '}_{' + idx[1][i] + '=0}'
    first_loops += 'B^{' + str(op_inputs_shape[0][dim]) + '}_{tx=0}'
    second_loops += 'B^{' + str(op_inputs_shape[1][dim]) + '}_{tx=0}'
    for i in range(dim+1, index_len):
        first_loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[0][i] + '=0}'
        second_loops += 'L^{' + str(op_inputs_shape[1][i]) + '}_{' + idx[1][i] + '=0}'
    if first_loops=='':
        first_loops='B^{1}_{tx=0}'
    if second_loops=='':
        second_loops='B^{1}_{tx=0}'
    IR = first_loops + '[' + var_outputs[0] + '=' + var_inputs[0] + ';];'
    IR += second_loops + '[' + var_outputs[1] + '=' + var_inputs[1] + ';];'
    return IR

def nll_loss_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #nll_loss has two inputs (input, target) and one output
    #kwargs: weight=None, reduction='mean', ignore_index=-100, label_smoothing=0.0
    #intermediate: weight_c_output, weight_c_input, l_n, sum_weight_c
    #we only consider weight is None
    has_weight=False
    reduction = op_kwargs['reduction'] if 'reduction' in op_kwargs else 'mean'
    ignore_index = op_kwargs['ignore_index'] if 'ignore_index' in op_kwargs else -100
    index_len = len(op_inputs_shape[1])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[1][0]) + idx
    first_weight_idx = generate_loop_bind(op_inputs_shape[0][1])
    target_idx= [idx[0]]+idx[1:-1]
    var_targets = generate_var_IR(op_inputs[1], target_idx, op_inputs_dtype[1], op_inputs_shape[1])
    prediction_idx = [idx[0]]+[var_targets]+idx[1:-1]
    second_weight_idx = [var_targets]
    var_outputs = generate_var_IR(op_outputs[0], torch.Size([]), op_outputs_dtype[0], op_outputs_shape[0])
    var_prediction = generate_var_IR(op_inputs[0], prediction_idx, op_inputs_dtype[0], op_inputs_shape[0])
    
    temp_intermediate_names, name_start_idx = generate_names(3, name_start_idx)
    temp_intermediate_shape = [[op_inputs_shape[1]], [op_inputs_shape[1]], [op_inputs_shape[1]], [torch.Size([])]]
    temp_intermediate_dtype = [[op_inputs_dtype[1]], [op_inputs_dtype[1]], [op_inputs_dtype[0]], [op_inputs_dtype[1]]]
    
    var_intermediates = [generate_var_IR(temp_intermediate_names[0], first_weight_idx, temp_intermediate_dtype[0][0], temp_intermediate_shape[0][0]),
                          generate_var_IR(temp_intermediate_names[0], second_weight_idx, temp_intermediate_dtype[1][0], temp_intermediate_shape[1][0]),
                          generate_var_IR(temp_intermediate_names[1], [idx[0]]+idx[1:-1], temp_intermediate_dtype[2][0], temp_intermediate_shape[2][0]),
                          generate_var_IR(temp_intermediate_names[2], [], temp_intermediate_dtype[3][0], temp_intermediate_shape[3][0])]
    weight_loops = 'B^{' + str(op_inputs_shape[0][1]) + '}_{tx=0}'
    main_loops = 'B^{' + str(op_inputs_shape[1][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        main_loops += 'L^{' + str(op_inputs_shape[1][i]) + '}_{' + idx[i] + '=0}'
    if has_weight:
        weight=generate_var_IR(op_inputs[2], first_weight_idx, op_inputs_dtype[2], op_inputs_shape[2])
        IR = weight_loops + '[' + var_intermediates[0] + '=if_then_else('+first_weight_idx[0]+'!='+str(ignore_index)+',1,0)*' + weight + ';];'
    else:
        IR = weight_loops + '[' + var_intermediates[0] + '=if_then_else('+first_weight_idx[0]+'!='+str(ignore_index)+',1,0);];'
    IR+=main_loops + '[' + var_intermediates[2] + '=-' + var_intermediates[1] +'*'+var_prediction+ ';];'
    intermediate_names, intermediate_shape, intermediate_dtype = intermediate_info
    if reduction=='mean': 
        IR+= main_loops + '[' + var_intermediates[3] + '=' + var_intermediates[3]+'+'+var_intermediates[1]+ ';];'
        IR+=main_loops + '[' + var_outputs + '=' + var_outputs + '+' + var_intermediates[2]+'/'+ var_intermediates[3] + ';];'     
        intermediate_names += temp_intermediate_names
        intermediate_shape += temp_intermediate_shape
        intermediate_dtype += temp_intermediate_dtype
        return IR, [intermediate_names, intermediate_shape, intermediate_dtype], name_start_idx
    elif reduction=='sum':
        IR+=main_loops + '[' + var_outputs + '=' + var_outputs + '+' + var_intermediates[2] + ';];'
        intermediate_names += temp_intermediate_names[:-1]
        intermediate_shape += temp_intermediate_shape[:-1]
        intermediate_dtype += temp_intermediate_dtype[:-1]
        return IR, [intermediate_names, intermediate_shape, intermediate_dtype], name_start_idx-1
    
def cross_entropy_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #cross_entropy has two inputs (input, target, weight) and one output
    #kwargs: weight='None', reduction='mean', ignore_index=-100, label_smoothing=0.0
    #logsoftmax+nllloss
    reduction = op_kwargs['reduction'] if 'reduction' in op_kwargs else 'mean'
    ignore_index = op_kwargs['ignore_index'] if 'ignore_index' in op_kwargs else -100
    #logsoftmax
    temp_intermediate_names, name_start_idx = generate_names(1, name_start_idx)
    temp_intermediate_shape = [op_inputs_shape[0]]
    temp_intermediate_dtype = [op_inputs_dtype[0]]
    IR, intermediate_info, name_start_idx = log_softmax_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, temp_intermediate_names, temp_intermediate_shape, temp_intermediate_dtype, {'dim':1}, intermediate_info, name_start_idx)
    #nll_loss
    IRtemp, intermediate_info, name_start_idx = nll_loss_to_IR([temp_intermediate_names[0], op_inputs[1]], [temp_intermediate_shape[0], op_inputs_shape[1]], [temp_intermediate_dtype[0], op_inputs_dtype[1]], op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
    IR+= IRtemp
    return IR, intermediate_info, name_start_idx

def smooth_l1_loss_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    #smooth_l1_loss has two inputs (input, target) and one output
    #kwargs: beta=1.0, reduction='mean', size_average=None, reduce=None, batch_average=None
    #three intermediate vars
    beta = int(op_kwargs['beta']) if 'beta' in op_kwargs else 1.0
    reduction = op_kwargs['reduction'] if 'reduction' in op_kwargs else 'mean'
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    var_inputs =[generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0]),
                    generate_var_IR(op_inputs[1], idx, op_inputs_dtype[1], op_inputs_shape[1])]
    var_outputs = generate_var_IR(op_outputs[0], [], op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_names, name_start_idx = generate_names(4, name_start_idx)
    temp_intermediate_shape = [[op_inputs_shape[0]], [op_inputs_shape[0]], [op_inputs_shape[0]], [torch.Size([])]]
    temp_intermediate_dtype = [[op_inputs_dtype[0]], [op_inputs_dtype[0]], [op_inputs_dtype[0]], [op_inputs_dtype[0]]]
    var_intermediates = [generate_var_IR(temp_intermediate_names[0], idx, temp_intermediate_dtype[0][0], temp_intermediate_shape[0][0]),
                          generate_var_IR(temp_intermediate_names[1], idx, temp_intermediate_dtype[1][0], temp_intermediate_shape[1][0]),
                            generate_var_IR(temp_intermediate_names[2], idx, temp_intermediate_dtype[2][0], temp_intermediate_shape[2][0]),
                            generate_var_IR(temp_intermediate_names[3], [], temp_intermediate_dtype[3][0], temp_intermediate_shape[3][0])]
    #loops
    loops = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    second_loops = 'B^{1}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR= loops + '[' + var_intermediates[0] + '=0.5*(' + var_inputs[0] + '-' + var_inputs[1] +')**2/'+str(beta)+ ';];'
    IR+= loops + '[' + var_intermediates[1] + '=abs('+ var_inputs[0] + '-' + var_inputs[1] + ')-0.5*' + str(beta) + ';];' 
    IR+= loops + '[' + var_intermediates[2] + '=if_then_else(abs(' + var_inputs[0] + '-' + var_inputs[1] + ')<' + str(beta) + ',' + var_intermediates[0] + ',' + var_intermediates[1] + ');];'
    intermediate_names, intermediate_shape, intermediate_dtype = intermediate_info
    if reduction == 'mean':
        IR+=loops+ '[' + var_intermediates[3] + '=' + var_intermediates[3] + '+' + var_intermediates[2] + ';];'
        IR+=second_loops+'[' + var_outputs + '=' + var_intermediates[3] + '/' + str(torch.prod(torch.tensor(op_inputs_shape[0])).item())+ ';];'
        intermediate_names += temp_intermediate_names
        intermediate_shape += temp_intermediate_shape
        intermediate_dtype += temp_intermediate_dtype
        return IR, [intermediate_names,intermediate_shape, intermediate_dtype], name_start_idx
    elif reduction == 'sum':
        IR+=loops+ '[' + var_outputs + '=' + var_outputs + '+' + var_intermediates[2] + ';];'
        intermediate_names += temp_intermediate_names[:-1]
        intermediate_shape += temp_intermediate_shape[:-1]
        intermediate_dtype += temp_intermediate_dtype[:-1]
        return IR, [intermediate_names, intermediate_shape, intermediate_dtype], name_start_idx-1

def cosine_similarity_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    # cosine_similarity has two inputs (input1, input2) and one output
    # kwargs: dim=1, eps=1e-8, normalize=False
    # three intermediate vars: uv, |u|_2, |v|_2
    dim = op_kwargs['dim'] if 'dim' in op_kwargs else 1
    if dim<0:
        dim += len(op_inputs_shape[0])
    eps = op_kwargs['eps'] if 'eps' in op_kwargs else 1e-8
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    if dim!=0:
        idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    else:
        idx = idx+generate_loop_bind(op_inputs_shape[0][-1])
    var_inputs = [generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0]),
                    generate_var_IR(op_inputs[1], idx, op_inputs_dtype[1], op_inputs_shape[1])]
    var_outputs = generate_var_IR(op_outputs[0], idx[:dim]+idx[dim+1:], op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_names, name_start_idx = generate_names(3, name_start_idx)
    temp_shape=op_inputs_shape[0][:dim]+ op_inputs_shape[0][dim+1:]
    temp_intermediate_shape = [[temp_shape], [temp_shape], [temp_shape]]
    temp_intermediate_dtype = [[op_inputs_dtype[0]], [op_inputs_dtype[0]], [op_inputs_dtype[0]]]
    var_intermediates = [generate_var_IR(temp_intermediate_names[0], idx[:dim]+idx[dim+1:], temp_intermediate_dtype[0][0], temp_intermediate_shape[0][0]),
                          generate_var_IR(temp_intermediate_names[1], idx[:dim]+idx[dim+1:], temp_intermediate_dtype[1][0], temp_intermediate_shape[1][0]),
                          generate_var_IR(temp_intermediate_names[2], idx[:dim]+idx[dim+1:], temp_intermediate_dtype[2][0], temp_intermediate_shape[2][0])]
    #loops
    first_loops, second_loops='',''
    for i in range(1, dim):
        first_loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
        second_loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    first_loops += 'L^{' + str(op_inputs_shape[0][dim]) + '}_{' + idx[dim] + '=0}'
    for i in range(dim+1, index_len-1):
        first_loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
        second_loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    if dim!=0 and dim!=index_len-1:
        loop_bind = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
        first_loops = loop_bind + first_loops+'L^{' + str(op_inputs_shape[0][-1]) + '}_{' + idx[-1] + '=0}'
        second_loops = loop_bind + second_loops+'L^{' + str(op_inputs_shape[0][-1]) + '}_{' + idx[-1] + '=0}'
    elif dim== index_len-1:
        loop_bind = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
        first_loops = loop_bind + first_loops
        second_loops = loop_bind + second_loops
    else:
        loop_bind = 'B^{' + str(op_inputs_shape[0][-1]) + '}_{tx=0}'
        first_loops = 'L^{' + str(op_inputs_shape[0][0]) + '}_{' + idx[0] + '=0}' + first_loops + loop_bind
        second_loops = 'L^{' + str(op_inputs_shape[0][0]) + '}_{' + idx[0] + '=0}' + second_loops + loop_bind
    if first_loops=='':
        first_loops='B^{1}_{tx=0}'
    if second_loops=='':
        second_loops='B^{1}_{tx=0}'
    #mul
    IR = first_loops + '[' + var_intermediates[0] + '=' +var_intermediates[0]+'+' + var_inputs[0] + '*' + var_inputs[1] + ';];'
    #squared l2norm
    IR += first_loops + '[' + var_intermediates[1] + '=' + var_intermediates[1] + '+' + var_inputs[0] + '**2;];'
    IR += first_loops + '[' + var_intermediates[2] + '=' + var_intermediates[2] + '+' + var_inputs[1] + '**2;];'
    #final
    IR += second_loops+'['+var_outputs + '=' + var_intermediates[0] + '/max(sqrt(' + var_intermediates[1] + ')*sqrt(' + var_intermediates[2] + '),'+str(eps)+');];'
    return IR, [intermediate_info[0] + temp_intermediate_names, intermediate_info[1] + temp_intermediate_shape, intermediate_info[2] + temp_intermediate_dtype], name_start_idx

def kl_div_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    # kl_div has two inputs (input, target) and one output
    # kwargs: reduction='none', log_target=False
    # three intermediate vars: loss_pointwise, sum
    reduction = op_kwargs['reduction'] if 'reduction' in op_kwargs else 'none'
    log_target = op_kwargs['log_target'] if 'log_target' in op_kwargs else False
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = [idx[0]]+ generate_loop_bind(op_inputs_shape[0][1])+ idx[1:]
    var_inputs = [generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0]),
                    generate_var_IR(op_inputs[1], idx, op_inputs_dtype[1], op_inputs_shape[1])]
    var_outputs = generate_var_IR(op_outputs[0], [], op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_names, name_start_idx = generate_names(2, name_start_idx)
    temp_intermediate_shape = [[op_inputs_shape[0]], [torch.Size([])]]
    temp_intermediate_dtype = [[op_inputs_dtype[0]], [op_inputs_dtype[0]]]
    var_intermediates = [generate_var_IR(temp_intermediate_names[0], idx, temp_intermediate_dtype[0][0], temp_intermediate_shape[0][0]),
                          generate_var_IR(temp_intermediate_names[1], [], temp_intermediate_dtype[1][0], temp_intermediate_shape[1][0])]
    #loops
    loops = 'L^{' + str(op_inputs_shape[0][0]) + '}_{' + idx[0] + '=0}'+'B^{' + str(op_inputs_shape[0][1]) + '}_{tx=0}'
    second_loops = 'B^{1}_{tx=0}'
    for i in range(2, index_len):
        loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    if reduction != 'none':
        if not log_target:
            IR = loops + '[' + var_intermediates[0] + '=' + var_inputs[1] + '*(log(' + var_inputs[1] + ')-' + var_inputs[0] + ');];'
        else:
            IR = loops + '[' + var_intermediates[0] + '=exp(' + var_inputs[1] + ')*(' + var_inputs[1] + '-' + var_inputs[0] + ');];'
    else:
        if not log_target:
            IR = loops + '[' + var_outputs + '=' + var_inputs[1] + '*(log(' + var_inputs[1] + ')-' + var_inputs[0] + ');];'
        else:
            IR = loops + '[' + var_outputs + '=exp(' + var_inputs[1] + ')*(' + var_inputs[1] + '-' + var_inputs[0] + ');];'
        return IR, intermediate_info, name_start_idx
    if reduction == 'mean':
        IR += loops + '[' + var_intermediates[1] + '=' + var_intermediates[1] + '+' + var_intermediates[0] + ';];'
        IR += second_loops+'[' + var_outputs + '=' + var_intermediates[1] + '/' + str(torch.prod(torch.tensor(op_inputs_shape[0])).item())+ ';];'
        intermediate_names, intermediate_shape, intermediate_dtype = intermediate_info
        intermediate_names += temp_intermediate_names
        intermediate_shape += temp_intermediate_shape
        intermediate_dtype += temp_intermediate_dtype
        return IR, [intermediate_names, intermediate_shape, intermediate_dtype], name_start_idx
    elif reduction == 'batchmean':
        IR += loops + '[' + var_intermediates[1] + '=' + var_intermediates[1] + '+' + var_intermediates[0] + ';];'
        IR += second_loops+'[' + var_outputs + '=' + var_intermediates[1] + '/' + str(op_inputs_shape[0][0])+ ';];'
        intermediate_names, intermediate_shape, intermediate_dtype = intermediate_info
        intermediate_names += temp_intermediate_names
        intermediate_shape += temp_intermediate_shape
        intermediate_dtype += temp_intermediate_dtype
        return IR, [intermediate_names, intermediate_shape, intermediate_dtype], name_start_idx
    elif reduction == 'sum':
        IR += loops + '[' + var_outputs + '=' + var_outputs + '+' + var_intermediates[0] + ';];'
        intermediate_names, intermediate_shape, intermediate_dtype = intermediate_info
        intermediate_names += temp_intermediate_names[:-1]
        intermediate_shape += temp_intermediate_shape[:-1]
        intermediate_dtype += temp_intermediate_dtype[:-1]
        return IR, [intermediate_names, intermediate_shape, intermediate_dtype], name_start_idx-1

def triplet_margin_loss_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx):
    # triplet_margin_loss has three inputs (anchor, positive, negative) and one output
    # kwargs: margin=1.0, p=2, eps=1e-6, swap=False, reduction='mean'
    # four intermediate vars: norm_ap, norm_an, loss, loss_sum
    margin = float(op_kwargs['margin']) if 'margin' in op_kwargs else 1.0
    p = int(op_kwargs['p']) if 'p' in op_kwargs else 2
    eps = float(op_kwargs['eps']) if 'eps' in op_kwargs else 1e-6
    reduction = op_kwargs['reduction'] if 'reduction' in op_kwargs else 'mean'
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    var_inputs = [generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0]),
                    generate_var_IR(op_inputs[1], idx, op_inputs_dtype[1], op_inputs_shape[1]),
                    generate_var_IR(op_inputs[2], idx, op_inputs_dtype[2], op_inputs_shape[2])]
    var_outputs = generate_var_IR(op_outputs[0], [], op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_names, name_start_idx = generate_names(4, name_start_idx)
    temp_intermediate_shape = [[op_inputs_shape[0]], [op_inputs_shape[0]], [op_inputs_shape[0]], [torch.Size([])]]
    temp_intermediate_dtype = [[op_inputs_dtype[0]], [op_inputs_dtype[0]], [op_inputs_dtype[0]], [op_inputs_dtype[0]]]
    var_intermediates = [generate_var_IR(temp_intermediate_names[0], [idx[0]], temp_intermediate_dtype[0][0], temp_intermediate_shape[0][0]),
                          generate_var_IR(temp_intermediate_names[1], [idx[0]], temp_intermediate_dtype[1][0], temp_intermediate_shape[1][0]),
                          generate_var_IR(temp_intermediate_names[2], [idx[0]], temp_intermediate_dtype[2][0], temp_intermediate_shape[2][0]),
                          generate_var_IR(temp_intermediate_names[3], [], temp_intermediate_dtype[3][0], temp_intermediate_shape[3][0])]
    #loops
    loops = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    max_loops = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    second_loops = 'B^{1}_{tx=0}'
    #norm
    if p>1:
        IR = loops + '[' + var_intermediates[0] + '=' +var_intermediates[0]+'+('+ var_inputs[0] +'-'+var_inputs[1]+ ')**' + str(p) + ';];'
        IR += loops + '[' + var_intermediates[1] + '=' +var_intermediates[1]+'+('+ var_inputs[0] +'-'+var_inputs[2]+ ')**' + str(p) + ';];'
        IR+= max_loops + '[' + var_intermediates[2] + '=max(' + var_intermediates[0] +'**(1/'+str(p)+ ')-' + var_intermediates[1] + '**(1/'+str(p)+ ')+' + str(margin) + ',0);];'
    elif p==1:
        IR = loops + '[' + var_intermediates[0] + '=' +var_intermediates[0]+'+abs(' + var_inputs[0] +'-'+var_inputs[1]+ ');];'
        IR += loops + '[' + var_intermediates[1] + '=' +var_intermediates[1]+'+abs(' + var_inputs[0] +'-'+var_inputs[2]+ ');];'
        IR+= max_loops + '[' + var_intermediates[2] + '=max(' + var_intermediates[0] +' -' + var_intermediates[1] + '+' + str(margin) + ',0);];'
    if reduction == 'mean':
        IR += max_loops + '[' + var_intermediates[3] + '=' + var_intermediates[3] + '+' + var_intermediates[2] + ';];'
        IR += second_loops+'[' + var_outputs + '=' + var_intermediates[3] + '/' + str(op_inputs_shape[0][0])+ ';];'
        intermediate_names, intermediate_shape, intermediate_dtype = intermediate_info
        intermediate_names += temp_intermediate_names
        intermediate_shape += temp_intermediate_shape
        intermediate_dtype += temp_intermediate_dtype
        return IR, [intermediate_names, intermediate_shape, intermediate_dtype], name_start_idx
    elif reduction == 'sum':
        IR += max_loops + '[' + var_outputs + '=' + var_outputs + '+' + var_intermediates[2] + ';];'
        intermediate_names, intermediate_shape, intermediate_dtype = intermediate_info
        intermediate_names += temp_intermediate_names[:-1]
        intermediate_shape += temp_intermediate_shape[:-1]
        intermediate_dtype += temp_intermediate_dtype[:-1]
        return IR, [intermediate_names, intermediate_shape, intermediate_dtype], name_start_idx-1

def linear_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #linear has three inputs (input, weight, bias) and one output
    #output=input @ weight.t() + bias
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    input_idx=idx[:len(op_inputs_shape[0])-1]+[idx[-1]]
    weight_idx = [idx[-2]]+[idx[-1]]
    if op_inputs[2]=='None':
        var_inputs = [generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0]),
                        generate_var_IR(op_inputs[1], weight_idx, op_inputs_dtype[1], op_inputs_shape[1])]
    else:
        bias_idx = [idx[-2]]
        var_inputs = [generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0]),
                        generate_var_IR(op_inputs[1], weight_idx, op_inputs_dtype[1], op_inputs_shape[1]),
                        generate_var_IR(op_inputs[2], bias_idx, op_inputs_dtype[2], op_inputs_shape[2])]
    var_outputs = generate_var_IR(op_outputs[0], idx[:-1], op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    loops = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    second_loops= 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
        second_loops+='L^{' + str(op_outputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    loops += 'L^{' + str(op_inputs_shape[0][-1]) + '}_{' + idx[-1] + '=0}'
    if op_inputs[2]=='None':
        IR= loops + '[' + var_outputs + '=' + var_outputs+'+' + var_inputs[0] + '*' + var_inputs[1] + ';];'
    else:
        IR= loops + '[' + var_outputs + '=' + var_outputs+'+' + var_inputs[0] + '*' + var_inputs[1] + ';];'
        IR+= second_loops + '[' + var_outputs + '=' + var_outputs + '+' + var_inputs[2] + ';];'
    return IR

def split_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs):
    #split has two input (input, split size) and multiple outputs
    #kwargs: dim=0
    dim = op_kwargs['dim'] if 'dim' in op_kwargs else 0
    split_size = int(op_inputs[1])
    if dim<0:
        dim += len(op_inputs_shape[0])
    output_num = len(op_outputs_shape[0])
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    if dim!=0:
        idx = generate_loop_bind(op_outputs_shape[0][0][0]) + idx
        loops= 'B^{' + str(op_outputs_shape[0][0][0]) + '}_{tx=0}'
        for i in range(1,index_len):
            loops += 'L^{' + str(op_outputs_shape[0][0][i]) + '}_{' + idx[i] + '=0}'
    else:
        idx = idx + generate_loop_bind(op_inputs_shape[0][0][-1])
        loops=''
        for i in range(0, index_len-1):
            loops += 'L^{' + str(op_inputs_shape[0][0][i]) + '}_{' + idx[i] + '=0}'
        loops+= 'B^{' + str(op_inputs_shape[0][0][-1]) + '}_{tx=0}'
    var_inputs = []
    input_idx=[idx[:dim]+[idx[dim]]+idx[dim+1:]]
    var_inputs= [generate_var_IR(op_inputs[0], input_idx[0], op_inputs_dtype[0], op_inputs_shape[0])]
    var_outputs=[generate_var_IR(op_outputs[0], ['0']+idx, str(op_outputs_dtype[0][0]), op_outputs_shape[0][0])]
    if loops=='':
        loops='B^{1}_{tx=0}'
    IR= loops + '[' + var_outputs[0] + '=' + var_inputs[0] + ';];'
    for i in range(1, output_num):
        input_idx.append(idx[:dim]+[idx[dim]+'+'+str(i*split_size)]+idx[dim+1:])
        var_inputs.append(generate_var_IR(op_inputs[0], input_idx[i], op_inputs_dtype[0], op_inputs_shape[0]))
        var_outputs.append(generate_var_IR(op_outputs[0], [str(i)]+idx, str(op_outputs_dtype[0][i]), op_outputs_shape[0][0]))
        IR += loops + '[' + var_outputs[i] + '=' + var_inputs[i] + ';];'
    return IR

def view_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #view has multiple inputs (input, shape) and one output
    if len(op_inputs_shape[0])>=len(op_outputs_shape[0]):
        idx, loops, output_idx=view_info(op_inputs_shape[0], op_outputs_shape[0])
    else:
        output_idx, loops, idx = view_info(op_outputs_shape[0], op_inputs_shape[0])
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    if loops=='':
        loops='B^{1}_{tx=0}'
    IR = loops + '[' + var_outputs + '=' + var_inputs + ';];'
    return IR

def expand_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #expand has multiple inputs (input, shape) and one output
    input_shape=op_inputs_shape[0]
    output_shape=op_outputs_shape[0]
    diff_idx = [i for i, (ai, bi) in enumerate(zip(input_shape, output_shape)) if ai != bi]
    index_len = len(op_outputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    # print(f'expand: input_shape:{input_shape}, output_shape:{output_shape}, diff_idx:{diff_idx}')
    if diff_idx==0:
        input_idx=['0']+idx
        output_idx=idx+['tx']
        loops = ''
        for i in range(index_len-1):
            loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + output_idx[i] + '=0}'
        loops = 'B^{' + str(op_outputs_shape[0][index_len-1]) + '}_{tx=0}'
    else:
        output_idx=['tx']+idx
        input_idx=output_idx.copy()
        input_idx[diff_idx[0]]='0'
        loops = 'B^{' + str(op_outputs_shape[0][0]) + '}_{tx=0}'
        for i in range(1, index_len):
            loops += 'L^{' + str(op_outputs_shape[0][i]) + '}_{' + output_idx[i] + '=0}'
    var_inputs = generate_var_IR(op_inputs[0], input_idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    # print(f'expand: var_inputs:{var_inputs}, var_outputs:{var_outputs}')
    if loops=='':
        loops='B^{1}_{tx=0}'
    IR = loops + '[' + var_outputs + '=' + var_inputs + ';];'
    # print(f'expand: IR:{IR}')
    return IR

def transpose_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #transpose has two inputs (input, dim1, dim2) and one output
    dim1 = int(op_inputs[1])
    if dim1<0:
        dim1 += len(op_inputs_shape[0])
    dim2 = int(op_inputs[2])
    if dim2<0:
        dim2 += len(op_inputs_shape[0])
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    if dim1<dim2:
        output_idx = idx[:dim1] + [idx[dim2]] + idx[dim1+1:dim2] + [idx[dim1]] + idx[dim2+1:]
    else:
        output_idx = idx[:dim2] + [idx[dim1]] + idx[dim2+1:dim1] + [idx[dim2]] + idx[dim1+1:]
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], output_idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    loops = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR = loops + '[' + var_outputs + '=' + var_inputs + ';];'
    return IR

def size_to_IR(op_inputs, op_inputs_shape, op_outputs, op_outputs_shape, op_outputs_dtype):
    #size has two inputs (input, dim) and one output
    if len(op_inputs)==2:
        dim = int(op_inputs[1])
        if dim<0:
            dim += len(op_inputs_shape[0])
        output_dtype='FLOAT64'
        var_outputs = generate_var_IR(op_outputs[0], [], output_dtype, op_outputs_shape[0])
        IR = 'B^{1}_{tx=0}[' + var_outputs + '=' + str(op_inputs_shape[0][dim]) + ';];'
        return IR

def masked_fill_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #masked_fill has three inputs (input, mask, value) and one output
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    if len(op_inputs_shape[0])==len(op_inputs_shape[1]) and op_inputs_shape[0]==op_inputs_shape[1]:
        mask_idx=idx
    elif len(op_inputs_shape[0])==len(op_inputs_shape[1]) and op_inputs_shape[0]!=op_inputs_shape[1]:
        mask_idx = []
        for i in range(len(op_inputs_shape[0])):
            if op_inputs_shape[1][i]==1:
                mask_idx.append('0')
            elif op_inputs_shape[1][i]==op_inputs_shape[0][i]:
                mask_idx.append(idx[i])
    elif op_inputs_shape[0]!=op_inputs_shape[1]:
        start_input=0
        mask_idx = []
        for i in range(len(op_inputs_shape[1])):
            mask_dim = op_inputs_shape[1][i]
            for j in range(start_input, len(op_inputs_shape[0])):
                input_dim = op_inputs_shape[0][j]
                if mask_dim==input_dim:
                    mask_idx.append(idx[j])
                    start_input = j+1
                    break
                elif mask_dim==1:
                    mask_idx.append('0')
                    start_input = j
                    break
    var_inputs = [generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0]),
                    generate_var_IR(op_inputs[1], mask_idx, op_inputs_dtype[1], op_inputs_shape[1]),
                    generate_var_IR(op_inputs[2], [], op_inputs_dtype[2], op_inputs_shape[2])]
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    loops = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR = loops + '[' + var_outputs + '=if_then_else(' + var_inputs[1] + ',' + var_inputs[2] + ', ' + var_inputs[0] + ');];'
    return IR

def direct_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    loops = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR = loops + '[' + var_outputs + '=' + var_inputs + ';];'
    return IR

def mish_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, intermediate_info, name_start_idx):
    #mish has one input (input) and one output
    # intermediate vars: softplus_output, tanh_output
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    temp_intermediate_names, name_start_idx = generate_names(2, name_start_idx)
    temp_intermediate_shape = [[op_inputs_shape[0]], [op_inputs_shape[0]]]
    temp_intermediate_dtype = [[op_inputs_dtype[0]], [op_inputs_dtype[0]]]
    var_intermediates = generate_var_IR(temp_intermediate_names[1], idx, temp_intermediate_dtype[1][0], temp_intermediate_shape[1][0])
    #loops
    loops = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    #softplus
    IR=softplus_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, [temp_intermediate_names[0]], temp_intermediate_shape[0], temp_intermediate_dtype[0])
    #tanh
    IRtemp, intermediate_info, name_start_idx=tanh_to_IR([temp_intermediate_names[0]], temp_intermediate_shape[0], temp_intermediate_dtype[0], [temp_intermediate_names[1]], temp_intermediate_shape[1], temp_intermediate_dtype[1], intermediate_info, name_start_idx)
    IR+=IRtemp
    #final
    IR += loops + '[' + var_outputs + '=' + var_inputs + '*' + var_intermediates + ';];'
    return IR, [intermediate_info[0] + temp_intermediate_names, intermediate_info[1] + temp_intermediate_shape, intermediate_info[2] + temp_intermediate_dtype], name_start_idx

def hardswish_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype):
    #hardswish has one input (input) and one output
    index_len = len(op_inputs_shape[0])
    idx = generate_idx_names(index_len-1, 0)
    idx = generate_loop_bind(op_inputs_shape[0][0]) + idx
    var_inputs = generate_var_IR(op_inputs[0], idx, op_inputs_dtype[0], op_inputs_shape[0])
    var_outputs = generate_var_IR(op_outputs[0], idx, op_outputs_dtype[0], op_outputs_shape[0])
    #loops
    loops = 'B^{' + str(op_inputs_shape[0][0]) + '}_{tx=0}'
    for i in range(1, index_len):
        loops += 'L^{' + str(op_inputs_shape[0][i]) + '}_{' + idx[i] + '=0}'
    IR = loops + '[' + var_outputs + '=' + var_inputs + '*max(min(' + var_inputs + '+3, 6), 0)/6;];'
    return IR

def handle_each_op(IR, op, len_ops, inputs_info, outputs_info, intermediate_info, constant_info, params_info, name_start_idx, last_mean_info=[], has_min_max=False):
    op_name, op_inputs, op_outputs, op_kwargs = op['name'], op['inputs'], op['outputs'], op['kwargs']
    op_inputs_shape, op_inputs_dtype, op_outputs_shape, op_outputs_dtype = loopup_variables(op_inputs, op_outputs, inputs_info, outputs_info, intermediate_info, constant_info, params_info)
    print(f'op_name:{op_name}')
    # print(f"op_inputs: {op_inputs}, op_inputs_shape: {op_inputs_shape}, op_inputs_dtype: {op_inputs_dtype}, op_output:{op_outputs}. op_outputs_shape: {op_outputs_shape}, op_outputs_dtype: {op_outputs_dtype}")
    len_ops-=1
    # print(f'len_ops:{len_ops}')
    if op_name=='matmul':
        IR+=matmul_to_IR(op_inputs,op_inputs_shape, op_inputs_dtype, op_outputs,op_outputs_shape, op_outputs_dtype)
    elif op_name=='einsum':
        IR+=einsum_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='softmax':
        IRtemp, intermediate_info, name_start_idx = softmax_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='mul' or op_name=='multiply' or op_name=='add' or op_name=='sub' or op_name=='truediv':
        IR+=mul_add_sub_truediv_to_IR(op_name,op_inputs,op_inputs_shape, op_inputs_dtype, op_outputs,op_outputs_shape, op_outputs_dtype)
    elif op_name=='clamp':
        IR+=clamp_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs)
    elif op_name=='mean':
        IRtemp, intermediate_info, name_start_idx = mean_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='diag':
        IR+=diag_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='getattr':
        if op_inputs[1]=='op_transpose':
            IR+=getattr_transpose_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='triu':
        IR+=triu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='tril':
        IR+=tril_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='relu':
        IR+=relu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='leaky_relu':
        IR+=leaky_relu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs)
    elif op_name=='sigmoid':
        IR+=sigmoid_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='tanh':
        IRtemp, intermediate_info, name_start_idx=tanh_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='log_softmax':
        IRtemp, intermediate_info, name_start_idx=log_softmax_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='gelu':
        IRtemp, intermediate_info, name_start_idx=gelu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='selu':
        IR+=selu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='hardsigmoid':
        IR+=hardsigmoid_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='softplus':
        IR+=softplus_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='abs' or op_name=='sqrt' or op_name=='log' or op_name=='exp':
        IR+=abs_sqrt_log_exp_to_IR(op_name, op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='elu':
        IR+= elu_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs)
    elif op_name=='hardtanh':
        IR+=hardtanh_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs)
    elif op_name=='running_mean':
        op_kwargs={'dim': tuple(x for x in range(len(op_inputs_shape[0])) if x!=1)}
        op_outputs_shape = [torch.Size([op_inputs_shape[0][1]])]
        op_outputs_dtype = [op_inputs_dtype[0]]
        last_mean_info=[op_outputs, op_outputs_shape, op_outputs_dtype]
        IRtemp, intermediate_info, name_start_idx = mean_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='running_var':
        op_kwargs={'dim': tuple(x for x in range(len(op_inputs_shape[0])) if x!=1)}
        op_kwargs['unbiased']=False
        op_inputs.extend(last_mean_info[0])
        op_inputs_shape.extend(last_mean_info[1])
        op_inputs_dtype.extend(last_mean_info[2])
        op_outputs_shape = [torch.Size([op_inputs_shape[0][1]])]
        op_outputs_dtype = [op_inputs_dtype[0]]
        IRtemp, intermediate_info, name_start_idx = running_var_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
        last_mean_info=[]
    elif op_name=='batch_norm':
        op_inputs_shape[1]=torch.Size([op_inputs_shape[0][1]])
        op_inputs_shape[2]=torch.Size([op_inputs_shape[0][1]])
        op_inputs_dtype[1]=op_inputs_dtype[0]
        op_inputs_dtype[2]=op_inputs_dtype[0]
        IRtemp, intermediate_info, name_start_idx = batch_norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='instance_norm':
        IRtemp, intermediate_info, name_start_idx = instance_norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='group_norm':
        IRtemp, intermediate_info, name_start_idx = group_norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='pow':
        IR+=pow_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='norm':
        IRtemp, intermediate_info, name_start_idx = norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='sum' or op_name=='prod':
        IRtemp, intermediate_info, name_start_idx=sum_prod_to_IR(op_name, op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name== 'bmm':
        IR+=bmm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='layer_norm':
        IRtemp, intermediate_info, name_start_idx = layer_norm_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='boolean_dispatch':
        IRtemp, intermediate_info, name_start_idx = boolean_dispatch_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='avg_pool':
        IRtemp, intermediate_info, name_start_idx = avg_pool_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='max' or op_name=='min':
        IRtemp, intermediate_info, name_start_idx = max_min_to_IR(op_name,op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='argmax' or op_name=='argmin':
        IRtemp, intermediate_info, name_start_idx = argmax_argmin_to_IR(op_name, op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='getitem':
        if not has_min_max:
            IR+=getitem_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='conv':
        IRtemp, intermediate_info, name_start_idx = conv_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='cumsum' or op_name=='cumprod':
        IR+=cumsum_to_IR(op_name, op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs)
    elif op_name=='flip':
        IR+=flip_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='select':
        IR+=select_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='unsqueeze':
        IR+=unsqueeze_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='squeeze':
        IR+=squeeze_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='zeros_like':
        IR+=zeros_like_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='cat':
        IR+= cat_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype,op_kwargs)
    elif op_name=='cross_entropy':
        IRtemp, intermediate_info, name_start_idx = cross_entropy_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='nll_loss':
        IRtemp, intermediate_info, name_start_idx = nll_loss_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='smooth_l':
        IRtemp, intermediate_info, name_start_idx = smooth_l1_loss_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='cosine_similarity':
        IRtemp, intermediate_info, name_start_idx = cosine_similarity_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='kl_div':
        IRtemp, intermediate_info, name_start_idx = kl_div_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='triplet_margin_loss':
        IRtemp, intermediate_info, name_start_idx = triplet_margin_loss_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='linear':
        IR+=linear_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='split':
        IR+=split_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs)
    elif op_name=='view':
        IR+=view_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='expand':
        IR+=expand_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='transpose':
        IR+=transpose_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='size':
        IR+=size_to_IR(op_inputs, op_inputs_shape, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='masked_fill':
        IR+=masked_fill_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='logsumexp':
        IRtemp, intermediate_info, name_start_idx=logsumexp_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, op_kwargs, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='mish':
        IRtemp, intermediate_info, name_start_idx=mish_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype, intermediate_info, name_start_idx)
        IR+=IRtemp
    elif op_name=='dropout' or op_name=='contiguous':
        IR+=direct_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    elif op_name=='hardswish':
        IR+=hardswish_to_IR(op_inputs, op_inputs_shape, op_inputs_dtype, op_outputs, op_outputs_shape, op_outputs_dtype)
    else:
        len_ops+=1
        ValueError(f"Unsupported operation: {op_name}")
    # print(f"{op_name} op: {IR}")
    IR=IR.replace(' ','')
    if op_name=='max' or op_name=='min':
        has_min_max=True
    else:
        has_min_max=False
    # print(f'IR: {IR}')
    return IR,name_start_idx, last_mean_info, len_ops, has_min_max
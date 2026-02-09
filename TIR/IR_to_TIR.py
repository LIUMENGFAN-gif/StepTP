from tvm.script import from_source
import re
import tvm
from .utils import *

loop_notation=['L', 'P', 'V', 'B', 'U']

def add_vars(known_names, known_shapes, dtype_list):
    vars_string=""
    for idx in range(len(known_names)):
        var_dtype_string=dtype_list[idx]
        var_shape_string=str(tuple(known_shapes[idx]))
        vars_string+=known_names[idx]+": T.Buffer("+var_shape_string+", \""+var_dtype_string+"\")"
        if idx<len(known_names)-1:
            vars_string+=", "
        else:
            vars_string+="):\n"
    return vars_string


def add_body(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names, tab_num):
    body_string=[]
    inter_idx,inter_name,inter_shape=[],[],[]
    temp_tab_num=tab_num
    inter_dict={}
    output_dict={}
    ir_split_list=split_ir(ir)
    weird_sign=[]
    shape_correct_total=True
    # print(f'ir_split_list: {ir_split_list}')
    for ir_split_idx in range(len(ir_split_list)):
        ir_split=ir_split_list[ir_split_idx]
        # print(f'ir_split: {ir_split}')
        inter_dict, inter_string_list, inter_string_name, inter_string_shape, shape_correct=handle_intermediate_vars(ir_split, inter_names, inter_dict, ir_split_list[:ir_split_idx], tab_num)
        if not shape_correct:
            shape_correct_total=False
        body_string.extend(inter_string_list)
        inter_name.extend(inter_string_name)
        inter_idx.extend(list(range(len(body_string)-len(inter_string_name), len(body_string))))
        inter_shape.extend(inter_string_shape)
        # print(f'before inter_idx: {inter_idx}, inter_name: {inter_name},inter_shape:{inter_shape}')
        body_string, inter_name, inter_idx, inter_shape=check_repeated_inter(body_string, inter_name, inter_idx, inter_shape)
        # print(f'after inter_idx: {inter_idx}, inter_name: {inter_name},inter_shape:{inter_shape}')
        temp_tab_num=tab_num
        ir_split_string=[]
        while len(ir_split)>0:
            if ir_split[0] in loop_notation: #normal loop
                ir_split, loop_string, temp_tab_num, _=get_loops(ir_split, temp_tab_num)
                ir_split_string.append(loop_string)
            elif ir_split[0]=='[' and ir_split[1] in loop_notation:
                ir_split, loop_string, temp_tab_num, _=get_loops(ir_split[1:], temp_tab_num)
                ir_split_string.append(loop_string)
            elif ir_split[0]=='[' and ir_split[1] not in loop_notation:
                ir_split, compute_string, output_dict=get_compute(ir_split_list[:ir_split_idx], ir_split[1:], inter_dict, output_dict, known_names, known_shapes, known_dtype, input_known_names, temp_tab_num)
                ir_split_string.append(compute_string)
            elif ir_split[:2]=='];':
                ir_split=ir_split[2:]
            elif ir_split[0]==']':
                ir_split=ir_split[1:]
            elif ir_split[0]!='[' and ir_split[0] not in loop_notation and ir_split[:2]!='];':
                ir_split, compute_string, output_dict=get_compute(ir_split_list[:ir_split_idx], ir_split, inter_dict, output_dict, known_names, known_shapes, known_dtype, input_known_names, temp_tab_num)
                ir_split_string.append(compute_string)
            else:
                weird_sign.append(ir_split[0])
                ir_split= ir_split[1:]
        body_string.append(''.join(ir_split_string))
    return ''.join(body_string), weird_sign, shape_correct_total

def ir_to_tirstring(ir, known_names, known_shapes, known_dtype, input_known_names,inter_names):
    dtype_list=convert_torch_dtype(known_dtype)
    tab_num=1
    tir_string="# from tvm.script import ir as I\n# from tvm.script import tir as T\n"
    tir_string+="@I.ir_module\nclass Module:\n"
    tir_string+="\t"*tab_num+"@T.prim_func\n"+"\t"*tab_num+"def main("
    vars_string=add_vars(known_names, known_shapes, dtype_list)
    tir_string+=vars_string
    tab_num=2
    tir_string+="\t"*tab_num+"T.func_attr({\"tir.noalias\": True})\n"
    tir_string+="\t"*tab_num+"# with T.block(\"root\"):\n"
    body_string, weird_sign, shape_correct_total=add_body(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names, tab_num)
    tir_string+=body_string
    return tir_string, weird_sign, shape_correct_total

def build_tir_module(ir, known_names, known_shapes, known_dtype, input_known_names, target):
    inter_names=get_inter_info(ir, known_names)
    try:
        tir_string, weird_sign, shape_correct_total=ir_to_tirstring(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names)
        # print(f'tir_string:\n{tir_string}\nweird_sign:{weird_sign}\nshape_correct:{shape_correct_total}')
        if len(weird_sign)==0 and shape_correct_total:
            print("Building TIR module...")
            tvm_string = tvm.runtime.container.String(tir_string)
            print(f"TIR string is ready, converting to TIR module...")
            irmodule = from_source(tvm_string)
            try:
                target="cuda"
                print(f"Compiling TIR module...")
                f = tvm.build(irmodule, target=tvm.target.Target(target))
                print("TIR module compiled.")
                del tvm_string
                del irmodule
                print("successfully built TIR module.")
                return f, tir_string, True
            except Exception as e:
                error_info=f"Error in building TIR module: {e}"
                print(error_info)
                return error_info,'', False
        elif len(weird_sign)>0:
            error_info=f"Error in weird IR: {weird_sign}"
            return error_info,'', False
        elif not shape_correct_total:
            error_info="Error in shape correctness of IR."
            return error_info, '', False
    except Exception as e:
        error_info=f"Error in producing TIR string: {e}"
        return error_info, '', False

def build_tir_module_for_check_pass(ir, known_names, known_shapes, known_dtype, input_known_names, target):
    inter_names=get_inter_info(ir, known_names)
    try:
        tir_string, weird_sign, shape_correct_total=ir_to_tirstring(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names)
        if len(weird_sign)==0 and shape_correct_total:
            print("Building TIR module...")
            tvm_string = tvm.runtime.container.String(tir_string)
            print(f"TIR string is ready, converting to TIR module...{tvm.cuda().exist}")
            irmodule = from_source(tvm_string)
            try:
                print(f"Compiling TIR module...{tvm.cuda().exist}")
                f = tvm.build(irmodule, target=target)
                print("TIR module compiled.")
                del tvm_string
                del irmodule
                del f
                del tir_string
                print("successfully built TIR module.")
                return "success", '', True
            except Exception as e:
                error_info=f"Error in building TIR module: {e}"
                print(error_info)
                return error_info,'', False
        elif len(weird_sign)>0:
            error_info=f"Error in weird IR: {weird_sign}"
            return error_info,'', False
        elif not shape_correct_total:
            error_info="Error in shape correctness of IR."
            return error_info, '', False
    except Exception as e:
        error_info=f"Error in producing TIR string: {e}"
        return error_info, '', False

def build_tir_module_for_multi_IRs(ir, known_names, known_shapes, known_dtype, input_known_names, target):
    inter_names=get_inter_info(ir, known_names)
    try:
        tir_string, weird_sign, shape_correct_total=ir_to_tirstring(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names)
        if len(weird_sign)==0 and shape_correct_total:
            # print("Building TIR module...")
            tvm_string = tvm.runtime.container.String(tir_string)
            # print(f"TIR string is ready, converting to TIR module...")
            irmodule = from_source(tvm_string)
            try:
                f = tvm.build(irmodule, target=target)
                # print("TIR module compiled.")
                # print(f"f module:{dir(f)}")
                # print(f"get source code:{f.inspect_source()}")
                del irmodule
                del f
                print("successfully built TIR module.")
                return tvm_string, True
            except Exception as e:
                error_info=f"Error in building TIR module: {e}"
                print(error_info)
                return error_info, False
        elif len(weird_sign)>0:
            error_info=f"Error in weird IR: {weird_sign}"
            return error_info, False
        elif not shape_correct_total:
            error_info="Error in shape correctness of IR."
            return error_info, False
    except Exception as e:
        error_info=f"Error in producing TIR string: {e}"
        return error_info, False

def build_tir_module_for_multi_IRs_with_CUDA(ir, known_names, known_shapes, known_dtype, input_known_names, target):
    inter_names=get_inter_info(ir, known_names)
    try:
        tir_string, weird_sign, shape_correct_total=ir_to_tirstring(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names)
        if len(weird_sign)==0 and shape_correct_total:
            # print("Building TIR module...")
            tvm_string = tvm.runtime.container.String(tir_string)
            # print(f"TIR string is ready, converting to TIR module...")
            irmodule = from_source(tvm_string)
            try:
                f = tvm.build(irmodule, target=target)
                cuda=f.imported_modules[0].get_source()
                del irmodule
                del f
                print("successfully built TIR module.")
                return tvm_string, cuda, True
            except Exception as e:
                error_info=f"Error in building TIR module: {e}"
                print(error_info)
                return error_info, "", False
        elif len(weird_sign)>0:
            error_info=f"Error in weird IR: {weird_sign}"
            return error_info, "", False
        elif not shape_correct_total:
            error_info="Error in shape correctness of IR."
            return error_info, "", False
    except Exception as e:
        error_info=f"Error in producing TIR string: {e}"
        return error_info, "", False
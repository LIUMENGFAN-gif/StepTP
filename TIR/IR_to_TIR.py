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
    output_name=[name for name in known_names if name not in input_known_names][0]
    body_string=[]
    inter_idx,inter_name,inter_shape=[],[],[]
    calculated_output_shape=[]
    temp_tab_num=tab_num
    inter_dict={}
    output_dict={}
    #split the ir into different blocks
    ir_split_list=split_ir(ir)
    weird_sign=[]
    shape_correct_total=True
    # print(f'ir_split_list: {ir_split_list}\n')
    #handle each block
    for ir_split_idx in range(len(ir_split_list)):
        ir_split=ir_split_list[ir_split_idx]
        # print(f'ir_split: {ir_split}\n')
        #handle intermediate variables
        # print(f'inter_names: {inter_names}, inter_dict:{inter_dict}\n')
        inter_dict, inter_string_list, inter_string_name, inter_string_shape, shape_correct, output_shape_list=handle_intermediate_vars(ir_split, output_name, inter_names, inter_dict, ir_split_list[:ir_split_idx], known_names, known_shapes, tab_num)
        shape_correct=check_input_list(ir_split, inter_names, inter_dict, ir_split_list[:ir_split_idx], known_names, known_shapes)
        # print(f'inter_string_list:{inter_string_list}, shape_correct:{shape_correct}')
        # print(f'output_shape_list:{output_shape_list}')
        if not shape_correct:
            shape_correct_total=False
        body_string.extend(inter_string_list)
        inter_name.extend(inter_string_name)
        inter_idx.extend(list(range(len(body_string)-len(inter_string_name), len(body_string))))
        inter_shape.extend(inter_string_shape)
        calculated_output_shape, shape_correct_total=update_output_shape_list(output_shape_list, calculated_output_shape, shape_correct_total)
        # print(f'calculated_output_shape: {calculated_output_shape}')
        # print(f'before inter_idx: {inter_idx}, inter_name: {inter_name},inter_shape:{inter_shape}\n')
        body_string, inter_name, inter_idx, inter_shape=check_repeated_inter(body_string, inter_name, inter_idx, inter_shape)
        # print(f'after inter_idx: {inter_idx}, inter_name: {inter_name},inter_shape:{inter_shape}\n')
        list_eq_loops=handle_loop_nest_and_order(ir_split)
        # print(f'list_eq_loops: {list_eq_loops}\n')
        ir_split_string=[]
        for eq, full_loop_list, real_loop_list in list_eq_loops:
            temp_tab_num=tab_num+obtain_temp_tab_num(full_loop_list, real_loop_list)
            # print(f'tab_num:{tab_num}, temp_tab_num:{temp_tab_num}')
            for loop in real_loop_list:
                loop_string, temp_tab_num=get_loops_correct(loop, temp_tab_num)
                ir_split_string.append(loop_string)
            # print(f'after temp_tab_num:{temp_tab_num}')
            compute_string, output_dict=get_compute_correct(ir_split_list[:ir_split_idx], eq, inter_dict, output_dict, known_names, known_shapes, known_dtype, input_known_names, temp_tab_num)
            ir_split_string.append(compute_string)
        body_string.append(''.join(ir_split_string))
    return ''.join(body_string), weird_sign, shape_correct_total, calculated_output_shape

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
    body_string, weird_sign, shape_correct_total, calculated_output_shape=add_body(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names, tab_num)
    tir_string+=body_string
    return tir_string, weird_sign, shape_correct_total, calculated_output_shape

def build_tir_module(ir, known_names, known_shapes, known_dtype, input_known_names, target):
    inter_names=get_inter_info(ir, known_names)
    output_name=[name for name in known_names if name not in input_known_names][0]
    output_idx=known_names.index(output_name)
    output_shape=known_shapes[output_idx]
    # print(f'inter_names: {inter_names}, known_names:{known_names}, input_known_names:{input_known_names}, output_name:{output_name}, output_shape:{output_shape}\n')
    print("starting building tir string...")
    try:
        tir_string, weird_sign, shape_correct_total, calculated_output_shape=ir_to_tirstring(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names)
        print(f'tir_string:\n{tir_string}\nweird_sign:{weird_sign}\nshape_correct:{shape_correct_total}\ncalculated_output_shape:{calculated_output_shape}, output_shape:{output_shape}')
        print(len(weird_sign)==0 and shape_correct_total and calculated_output_shape==list(output_shape))
        if len(weird_sign)==0 and shape_correct_total and calculated_output_shape==list(output_shape):
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


def build_tir_module_for_multi_IRs(ir, known_names, known_shapes, known_dtype, input_known_names, target):
    inter_names=get_inter_info(ir, known_names)
    try:
        tir_string, weird_sign, shape_correct_total, calculated_output_shape=ir_to_tirstring(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names)
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
        tir_string, weird_sign, shape_correct_total, calculated_output_shape=ir_to_tirstring(ir, known_names, known_shapes, known_dtype, input_known_names, inter_names)
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

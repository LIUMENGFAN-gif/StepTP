import tvm
from .utils import *
import gc
import time
import torch

def verification(f, model, model_name, module, input_shapes, output_shape, output_dtype, target, constant_params_value, atol, rtol, dtype):
    if isinstance(f, str):
        print(f)
    else:
        print("start verification")
        ctx, device = generate_ctx_and_device(target)
        real_inputs=obtain_real_inputs_for_verification(module, model_name, input_shapes, device, dtype)
        if 'BatchNorm' in model_name:
            model.train()
        model = model.to(device)
        # print(f"real_inputs: {real_inputs}")
        torch_output= model(*real_inputs)
        # print(f"torch_output: {torch_output}")
        print("torch done")
        input_list=generate_f_input_list(ctx, real_inputs,constant_params_value, output_shape, output_dtype)
        # print(f'input_list: {input_list}')
        f(*input_list)
        tvm_output = input_list[-1].numpy()
        # print(f"tvm_output: {tvm_output}")
        print("tvm done")
        is_equal, info=check_if_two_outputs_equal(tvm_output, torch_output, atol=atol, rtol=rtol)#1e-3
        print(info)
        del input_list
        del real_inputs
        del torch_output
        del tvm_output
        del f
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        tvm.cuda(int(target[-1])).sync()
        return is_equal

def collect_verification(f, model, model_name, module, input_shapes, output_shape, output_dtype, target, constant_params_value, atol, rtol, dtype):
    if isinstance(f, str):
        print(f)
    else:
        print("start verification")
        ctx, device = generate_ctx_and_device(target)
        print(f'device:{device}, ctx:{ctx}')
        real_inputs=obtain_real_inputs_for_verification(module, model_name, input_shapes, device, dtype)
        print('real_inputs done')
        if 'BatchNorm' in model_name:
            model.train()
        print('input list')
        input_list=generate_f_input_list(ctx, real_inputs,constant_params_value, output_shape, output_dtype)
        # print("f codes:", f.imported_modules[0].get_source())
        f(*input_list)
        tvm_output = input_list[-1].numpy()
        print("tvm done")
        model = model.to(device)
        torch_output= model(*real_inputs)
        print("torch done")
        atol, rtol, MSE=check_two_outputs_precision_error(tvm_output, torch_output)
        print("start evaluation")
        evaluator=f.time_evaluator(f.entry_name, ctx, number=10, repeat=3)
        #warm-up
        for _ in range(3):
            evaluator(*input_list)
        prof_res=evaluator(*input_list)
        print(f'tvm mean:{prof_res.mean*1000:.4f}ms')
        #warm-up
        for _ in range(3):
            model(*real_inputs)
        torch.cuda.synchronize()
        start_event=torch.cuda.Event(enable_timing=True)
        end_event=torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(3):
            model(*real_inputs)
        end_event.record()
        torch.cuda.synchronize()
        curr_time=start_event.elapsed_time(end_event)/3
        print(f"torch mean:{curr_time}ms")
        return curr_time, prof_res.mean*1000, atol, rtol, MSE

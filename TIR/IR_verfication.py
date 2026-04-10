import tvm
from .utils import *
import gc
import time
import torch

def collect_verification(f, model, model_name, module, input_shapes, output_shape, output_dtype, target, constant_params_value, atol, rtol, dtype):
    if isinstance(f, str):
        print(f)
    else:
        ctx, device = generate_ctx_and_device(target)
        torch.cuda.set_device(device)
        real_inputs=obtain_real_inputs_for_verification(module, model_name, input_shapes, device, dtype)
        if 'BatchNorm' in model_name:
            model.train()
        input_list=generate_f_input_list(ctx, real_inputs,constant_params_value, output_shape, output_dtype)
        f(*input_list)
        tvm_output = input_list[-1]
        model = model.to(device)
        torch_output= model(*real_inputs)
        atol, rtol, MSE=check_two_outputs_precision_error(tvm_output, torch_output)
        evaluator=f.time_evaluator(f.entry_name, ctx, number=3, repeat=1)
        #warm-up
        for _ in range(3):
            evaluator(*input_list)
        prof_res=evaluator(*input_list)
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
        return curr_time, prof_res.mean*1000, atol, rtol, MSE

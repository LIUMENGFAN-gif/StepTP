import json
import tqdm
import ast
import difflib
import random
from TIR import *
from ops import *
import re
import os

strategy_dict={"operator_fusion": "The strategy operator fusion is used on the given IR to place consecutive equations from multiple similar loop nests into a single loop nest.",
"operator_fission": "The strategy operator fission is used on the given IR to split multiple equations in one loop nest into multiple separate loop nests.",
"compute_inline": "The strategy compute inline is used on the given IR to merge related equations from multiple loop nests into one equation within a single loop nest.",
"expression_splitting": "The strategy expression splitting is used on the given IR to separate a merged equation in one loop nest into multiple equations under multiple loop nests.",
"tensor_concat_to_fuse_operators": "The strategy tensor concat to fuse operators is used on the given IR to concatenate multiple input variables into one variable, merge the equations and then split the output variable to obtain multiple outputs.",
"tensor_split_to_decouple_operators": "The strategy tensor split to decouple operators is used on the given IR to split a input variable into multiple variables, run multiple equations and then concatenate multiple output variables into one variable.",
"common_subexpression_elimination": "The strategy common subexpression elimination is used on the given IR to compute duplicated expressions (i.e., expression=loop and equation) once and reuse the result to avoid redundant calculations.",
"expression_reorder": "The strategy expression reorder is used on the given IR to rearrange the expressions (i.e., expression=loop and equation).",
"loop_reorder": "The strategy loop reorder is used on the given IR to rearrange the nesting orders of loops within a loop nest.",
"loop_tiling": "The strategy loop tiling is used on the given IR to break two nested loops into four smaller loops (i.e., forming the tiles) within one loop nest.",
"loop_split": "The strategy loop split is used on the given IR to divide any loop within a loop nest into two nested loops.",
"loop_fusion": "The strategy loop fusion is used on the given IR to combine multiple loops within a loop nest into one loop.",
"loop_unrolling": "The strategy loop unrolling is used on the given IR to expand a loop body by computing its equation multiple times.",
"loop_parallelization": "The strategy loop parallelization is used on the given IR to distribute iterations of a loop across multiple processing units.",
"loop_vectorization": "The strategy loop vectorization is used on the given IR to transform loop operations to use SIMD instructions and thereby process multiple data elements simultaneously.",
"loop_binding": "The strategy loop binding is used on the given IR to map loop iterations to specific GPU threads or blocks along x y z axes with maximum 1024 1024 64 iterations per axis.",
"reduction_factorization": "The strategy reduction factorization is used on the given IR to restructure a reduction expression (i.e., expression=loop and equation) into multiple reduction expressions by splitting the reduction loop axis and then add the outputs.",
"cache_read_write": "The strategy cache read write is used on the given IR to move tensor variables between global (g), shared (s), and local (l) memory (i.e., shown in the superscript of the variable).",
"layout_transformation": "The strategy layout transformation is used on the given IR to change the memory arrangement of tensor variables (i.e., shown in the subscript of the variable).",
"set_storage_scope": "The strategy set storage scope is used on the given IR to directly set intermediate tensor variables between global (g), shared (s), and local (l) memory (i.e., shown in the superscript of the variable).",
"set_storage_layout": "The strategy set storage_layout is used on the given IR to directly change the memory arrangement of intermediate tensor variables (i.e., shown in the subscript of the variable).",
"precompute_indices": "The strategy precompute indices is used on the given IR to precompute indices of tensor variable (i.e., shown in the subscript of the variable) to store frequently used index expressions for reuse.",
"factorization": "The strategy factorization is used on the given IR to decompose a mathematical equation into products or sums of simpler components.",
"expand_factorization": "The strategy expand factorization is used on the given IR to reconstruct a factored equation into its original full equation (i.e., the reverse process of factorization).",
"cancellation": "The strategy cancellation is used on the given IR to remove variables that offset each other to simplify the equation.",
"expand_cancellation": "The strategy expand cancellation is used on the given IR to reconstruct the canceled variables to recover the original equation (i.e., the reverse process of cancellation).",
"apart": "The strategy apart is used on the given IR to decompose the rational fraction part into simpler partial fractions within one equation (i.e., the reverse process of together).", 
"together": "The strategy together is used on the given IR to combine multiple fractions into a single fraction within one equation (i.e., the reverse process of apart).",
"powsimp": "The strategy powsimp is used on the given IR to simplify the equation by combining and reducing power operations.",
"expand_powsimp": "The strategy expand powsimp is used on the given IR to reverse a simplified power operation into separate power operations within one equation (i.e., the reverse process of powsimp).",
"logsimp": "The strategy logsimp is used on the given IR to simplify logarithmic operations using log properties like product, quotient, and power rules within one equation.",
"expand_log": "The strategy expand log is used on the given IR to reverse the simplified logarithm operation into separate log operations within one equation (i.e., the reverse process of logsimp).",
"collect": "The strategy collect is used on the given IR to combine like operations into a simplified operation in an equation.",
"expand_collect": "The strategy expand collect is used on the given IR to split a combined like operation back into multiple operations in an equation (i.e., the reverse process of collect).",
"partially_equivalent_then_correct": "The strategy partially equivalent then correct is used on the given IR to first establish partial equivalence of several similar expressions (i.e., expression=loop and equation) by concatenating the input variables, computing the fused expression, and splitting the output variabls, as well as finally correct differences to achieve full equivalence by adding another expression.",
"exponential_split": "The strategy exponential split is used on the given IR to decompose an exponential operation into multiple factor operations by introducing an existing variable within one equation.",
"multiplicative_split": "The strategy multiplicative split is used on the given IR to decompose one operation into multiple multiplicative factor operations by introducing an existing variable within one equation.",
"additive_split": "The strategy additive split is used on the given IR to decompose one operation into multiple additive factor operations by introducing an existing variable within one equation.",
"normal_loop_max_to_prefix_max": "The strategy normal loop max to prefix max is used on the given IR to transform the maximum operations into online streaming operations where the current step is based on the previous step.",
"normal_loop_summation_on_exp_to_prefix_summation_on_exp": "The strategy normal loop summation to prefix summation on exp is used on the given IR to transform the summation of exponential operations into online streaming operations where the current step is based on the previous step. Note that this trick uses exponential cancellation to find the prefix relation.",
"online_softmax": "The strategy online softmax is used on the given IR to compute the softmax incrementally by updating the tensor variable step by step within a loop nested.",
"flashattention_wo_tiling": "The strategy flashattention without tiling is used on the given IR to compute attention expressions in an online manner by incrementally calculating scaled dot-products and applying online softmax.",
"normal_matmul_to_prefix_matmul_based_on_online_softmax": "The strategy normal matmul to prefix matmul based on online softmax is used on the given IR to transform a standard matrix multiplication into an online prefix computation based on online softmax applied in previous computations."}

def construct_operator_fusion_CoT(original_row_equations, transformed_row_equations, transformed_loops):
  multi_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  fused_expression=list(set(transformed_row_equations)-set(original_row_equations))
  loop=transformed_loops[transformed_row_equations.index(fused_expression[0])]
  # print(f'original_row_equations:{original_row_equations}\n transformed_row_equations:{transformed_row_equations}\n multi_expressions:{multi_expressions}')
  try:
    assert len(multi_expressions)==2
    assert len(fused_expression)==1
    assert loop in multi_expressions[0] and loop in multi_expressions[1]
    CoT_part=f" Under the given IR, the consecutive expressions \'{multi_expressions[0]}\' and \'{multi_expressions[1]}\' has the same loop nest \'{loop}\', and are fused into the expression \'{fused_expression[0]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["operator_fusion"]+CoT_part, True
  except:
    return "", False

def construct_operator_fission_CoT(original_row_equations, transformed_row_equations):
  multi_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  fused_expression=list(set(original_row_equations)-set(transformed_row_equations))
  try:
    assert len(multi_expressions)==2
    assert len(fused_expression)==1
    CoT_part=f" Under the given IR, \'{fused_expression[0]}\' with multiple equations are split into \'{multi_expressions[0]}\' and \'{multi_expressions[1]}\' as two expressions."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["operator_fission"]+CoT_part, True
  except:
    return "", False

def construct_compute_inline_CoT(original_row_equations, transformed_row_equations, transformed_loops):
  multi_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  fused_expression=list(set(transformed_row_equations)-set(original_row_equations))
  loop=transformed_loops[transformed_row_equations.index(fused_expression[0])]
  # print(f'original_row_equations:{original_row_equations}\n transformed_row_equations:{transformed_row_equations}\n multi_expressions:{multi_expressions}')
  try:
    assert len(multi_expressions)==2
    assert len(fused_expression)==1
    assert loop in multi_expressions[0] and loop in multi_expressions[1]
    CoT_part=f" Under the given IR, the expressions \'{multi_expressions[0]}\' and \'{multi_expressions[1]}\' has the same loop nest \'{loop}\' and related equations, and are merged into the expression \'{fused_expression[0]}\'."
    return strategy_dict["compute_inline"]+CoT_part, True
  except:
    return "", False

def construct_expression_splitting_CoT(original_row_equations, transformed_row_equations):
  multi_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  fused_expression=list(set(original_row_equations)-set(transformed_row_equations))
  try:
    assert len(multi_expressions)==2
    assert len(fused_expression)==1
    CoT_part=f" Under the given IR, \'{fused_expression[0]}\' with the merged equation are split into \'{multi_expressions[0]}\' and \'{multi_expressions[1]}\' as two expressions."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["expression_splitting"]+CoT_part, True
  except:
    return "", False

def construct_tensor_concat_to_fuse_operators_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  transformed_expressions=[item for item in transformed_row_equations if item in transformed_expressions]
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==2
    assert len(transformed_expressions)==5
    CoT_part=f" Under the given IR, for two similar expressions \'{original_expressions[0]}\' and \'{original_expressions[1]}\', the input variables are contenated by \'{transformed_expressions[0]}\' and \'{transformed_expressions[1]}\', the similar operations are executed by \'{transformed_expressions[2]}\', and the output variable is split into two outputs by \'{transformed_expressions[3]}\' and \'{transformed_expressions[4]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["tensor_concat_to_fuse_operators"]+CoT_part, True
  except:
    return "", False

def construct_tensor_split_to_decouple_operators_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  transformed_expressions=[item for item in transformed_row_equations if item in transformed_expressions]
  try:
    assert len(original_expressions)==1
    assert len(transformed_expressions)==6
    CoT_part=f" Under the given IR, for the expression \'{original_expressions[0]}\', the input variable is split into two inputs by \'{transformed_expressions[0]}\' and \'{transformed_expressions[1]}\', two similar operations are executed by \'{transformed_expressions[2]}\' and \'{transformed_expressions[3]}\', and two output variables are concatenated into one output by \'{transformed_expressions[4]}\' and \'{transformed_expressions[5]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["tensor_split_to_decouple_operators"]+CoT_part, True
  except:
    return "", False

def construct_common_subexpression_elimination_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  try:
    # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
    transformed_expressions_equation_part=[item[item.index('[')+1:-3].split("=")[1] for item in transformed_expressions]
    # print(f"transformed_expressions_equation_part:{transformed_expressions_equation_part}")
    common_part=[]
    for item_idx, item in enumerate(transformed_expressions_equation_part):
      add_item=True
      for original_item in original_expressions:
        if item not in original_item:
          add_item=False
      if add_item:
        common_part.append([item_idx, item])
    assert len(common_part)>0
    CoT_part=f" Under the given IR, a common equation part \'{common_part[0][1]}\' exists, so this part can be computed in the expression \'{transformed_expressions[common_part[0][0]]}\' once."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["common_subexpression_elimination"]+CoT_part, True
  except:
    return "", False

def construct_expression_reorder_CoT(original_row_equations, transformed_row_equations):
  try:
    assert set(original_row_equations)==set(transformed_row_equations)
    diff_expressions=[]
    for idx in range(len(original_row_equations)):
      if original_row_equations[idx]!=transformed_row_equations[idx]:
        diff_expressions.append(original_row_equations[idx])
    assert len(diff_expressions)>1
    CoT_part=f" Under the given IR, two expressions \'{diff_expressions[0]}\' and  \'{diff_expressions[1]}\' can be reordered."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["expression_reorder"]+CoT_part, True
  except:
    return "", False

def construct_loop_reorder_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops):
  # print(f"original_loops:{original_loops}\n transformed_loops:{transformed_loops}")
  try:
    assert len(original_loops)==len(transformed_loops)
    selected_loop=[]
    for idx in range(len(original_loops)):
      if original_loops[idx]!=transformed_loops[idx] and set(original_loops[idx])==set(transformed_loops[idx]):
        selected_loop.append([original_loops[idx], original_row_equations[idx],transformed_loops[idx],transformed_row_equations[idx]])
    # print(f"selected_loop:{selected_loop}")
    assert len(selected_loop)>0
    CoT_part=f" Under the given IR, the loops \'{selected_loop[0][0]}\' in the expression \'{selected_loop[0][1]}\' can be reordered as the loops \'{selected_loop[0][2]}\' in the expression \'{selected_loop[0][3]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["loop_reorder"]+CoT_part, True
  except:
    return "", False

def construct_loop_tiling_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops):
  try:
    assert len(original_loops)==len(transformed_loops)
    selected_loop=[]
    for idx in range(len(original_loops)):
      if original_loops[idx]!=transformed_loops[idx]:
        selected_loop.append([original_loops[idx], original_row_equations[idx],transformed_loops[idx],transformed_row_equations[idx]])
    # print(f"selected_loop:{selected_loop}")
    assert len(selected_loop)>0
    CoT_part=f" Under the given IR, the loops \'{selected_loop[0][0]}\' in the expression \'{selected_loop[0][1]}\' can be tiled as the loops \'{selected_loop[0][2]}\' in the expression \'{selected_loop[0][3]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["loop_tiling"]+CoT_part, True
  except:
    return "", False

def construct_loop_split_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops):
  try:
    assert len(original_loops)==len(transformed_loops)
    selected_loop=[]
    for idx in range(len(original_loops)):
      if original_loops[idx]!=transformed_loops[idx]:
        selected_loop.append([original_loops[idx], original_row_equations[idx],transformed_loops[idx],transformed_row_equations[idx]])
    # print(f"selected_loop:{selected_loop}")
    assert len(selected_loop)>0
    CoT_part=f" Under the given IR, the loops \'{selected_loop[0][0]}\' in the expression \'{selected_loop[0][1]}\' can be split as the loops \'{selected_loop[0][2]}\' in the expression \'{selected_loop[0][3]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["loop_split"]+CoT_part, True
  except:
    return "", False

def construct_loop_fusion_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops):
  try:
    assert len(original_loops)==len(transformed_loops)
    selected_loop=[]
    for idx in range(len(original_loops)):
      if original_loops[idx]!=transformed_loops[idx]:
        selected_loop.append([original_loops[idx], original_row_equations[idx],transformed_loops[idx],transformed_row_equations[idx]])
    # print(f"selected_loop:{selected_loop}")
    assert len(selected_loop)>0
    CoT_part=f" Under the given IR, the loops \'{selected_loop[0][0]}\' in the expression \'{selected_loop[0][1]}\' can be fused as the loops \'{selected_loop[0][2]}\' in the expression \'{selected_loop[0][3]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["loop_fusion"]+CoT_part, True
  except:
    return "", False

def construct_loop_unrolling_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops):
  try:
    assert len(original_loops)==len(transformed_loops)
    selected_loop=[]
    for idx in range(len(original_loops)):
      if original_loops[idx]!=transformed_loops[idx] and "U^" in transformed_loops[idx] :
        selected_loop.append([original_loops[idx], original_row_equations[idx],transformed_loops[idx],transformed_row_equations[idx]])
    # print(f"selected_loop:{selected_loop}")
    assert len(selected_loop)>0
    CoT_part=f" Under the given IR, one loop axis in the loops \'{selected_loop[0][0]}\' of the expression \'{selected_loop[0][1]}\' can be unrolled, then the loops become \'{selected_loop[0][2]}\' of the expression \'{selected_loop[0][3]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["loop_unrolling"]+CoT_part, True
  except:
    return "", False

def construct_loop_parallelization_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops):
  try:
    assert len(original_loops)==len(transformed_loops)
    selected_loop=[]
    for idx in range(len(original_loops)):
      if original_loops[idx]!=transformed_loops[idx] and "P^" in transformed_loops[idx] :
        selected_loop.append([original_loops[idx], original_row_equations[idx],transformed_loops[idx],transformed_row_equations[idx]])
    # print(f"selected_loop:{selected_loop}")
    assert len(selected_loop)>0
    CoT_part=f" Under the given IR, one loop axis in the loops \'{selected_loop[0][0]}\' of the expression \'{selected_loop[0][1]}\' can be parallel, then the loops become \'{selected_loop[0][2]}\' of the expression \'{selected_loop[0][3]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["loop_parallelization"]+CoT_part, True
  except:
    return "", False

def construct_loop_vectorization_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops):
  try:
    assert len(original_loops)==len(transformed_loops)
    selected_loop=[]
    for idx in range(len(original_loops)):
      if original_loops[idx]!=transformed_loops[idx] and "V^" in transformed_loops[idx] :
        selected_loop.append([original_loops[idx], original_row_equations[idx],transformed_loops[idx],transformed_row_equations[idx]])
    # print(f"selected_loop:{selected_loop}")
    assert len(selected_loop)>0
    CoT_part=f" Under the given IR, one loop axis in the loops \'{selected_loop[0][0]}\' of the expression \'{selected_loop[0][1]}\' can be vectorized, then the loops become \'{selected_loop[0][2]}\' of the expression \'{selected_loop[0][3]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["loop_vectorization"]+CoT_part, True
  except:
    return "", False

def construct_loop_binding_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops):
  try:
    assert len(original_loops)==len(transformed_loops)
    selected_loop=[]
    for idx in range(len(original_loops)):
      if original_loops[idx]!=transformed_loops[idx] and "B^" in transformed_loops[idx]:
        selected_loop.append([original_loops[idx], original_row_equations[idx],transformed_loops[idx],transformed_row_equations[idx]])
    # print(f"selected_loop:{selected_loop}")
    assert len(selected_loop)>0
    CoT_part=f" Under the given IR, one loop axis in the loops \'{selected_loop[0][0]}\' of the expression \'{selected_loop[0][1]}\' can be binded, then the loops become \'{selected_loop[0][2]}\' of the expression \'{selected_loop[0][3]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["loop_binding"]+CoT_part, True
  except:
    return "", False

def construct_reduction_factorization_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==1
    assert len(transformed_expressions)==3
    original_loop=original_loops[original_row_equations.index(original_expressions[0])]
    transformed_loop=[transformed_loops[transformed_row_equations.index(transformed_expressions[0])], transformed_loops[transformed_row_equations.index(transformed_expressions[1])]]
    CoT_part=f" Under the given IR, the reduction loop axis in the loops \'{original_loop}\' of the expression \'{original_expressions[0]}\' can be split, so the loops become \'{transformed_loop[0]}\' of the expression \'{transformed_expressions[0]}\' and \'{transformed_loop[1]}\' of the expression \'{transformed_expressions[1]}\'. Then the outputs are added by \'{transformed_expressions[2]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["reduction_factorization"]+CoT_part, True
  except:
    return "", False

def construct_cache_read_write_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  CoT_part=""
  try:
    assert len(original_expressions)==1
    assert len(transformed_expressions)==2
    original_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in original_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in original_expressions]
    original_variables=[itemitem for item in original_variables for itemitem in item]
    original_variables_filtered=list(set([item for item in original_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    transformed_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in transformed_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in transformed_expressions]
    transformed_variables=[itemitem for item in transformed_variables for itemitem in item]
    transformed_variables_filtered=list(set([item for item in transformed_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    moved_variable=[item for item in transformed_variables_filtered if ',l}' in item or ',s}' in item]
    # print(f"original_variables_filtered:{original_variables_filtered}, transformed_variables_filtered:{transformed_variables_filtered}, moved_variable:{moved_variable}")
    assert len(moved_variable)>0
    read, write=False, False
    read_variable, write_variable="",""
    for original_variable in original_variables_filtered:
      read_list=[True for item in transformed_expressions if moved_variable[0]+'='+original_variable+';' in item]
      write_list=[True for item in transformed_expressions if original_variable+'='+moved_variable[0]+';'  in item]
      read=True if True in read_list else False
      write=True if True in write_list else False
      read_variable=original_variable if read else ""
      write_variable=original_variable if write else ""
      if read or write:
        break
    if read:
      if ',l}' in moved_variable[0]:
        CoT_part=f" Under the given IR, the variable \'{read_variable}\' in the expression \'{original_expressions[0]}\' can be read into local memory as \'{moved_variable[0]}\' by two expressions \'{transformed_expressions[0]}\' and \'{transformed_expressions[1]}\'."
      elif ',s}' in moved_variable[0]:
        CoT_part=f" Under the given IR, the variable \'{read_variable}\' in the expression \'{original_expressions[0]}\' can be read into shared memory as \'{moved_variable[0]}\' by two expressions \'{transformed_expressions[0]}\' and \'{transformed_expressions[1]}\'."
    elif write:
      if ',l}' in moved_variable[0]:
        CoT_part=f" Under the given IR, the variable \'{write_variable}\' in the expression \'{original_expressions[0]}\' can write from local memory as \'{moved_variable[0]}\' by two expressions \'{transformed_expressions[0]}\' and \'{transformed_expressions[1]}\'."
      elif ',s}' in moved_variable[0]:
        CoT_part=f" Under the given IR, the variable \'{write_variable}\' in the expression \'{original_expressions[0]}\' can write from shared memory as \'{moved_variable[0]}\' by two expressions \'{transformed_expressions[0]}\' and \'{transformed_expressions[1]}\'."
    # print(f"CoT_part:{CoT_part}")
    if CoT_part!="":
      return strategy_dict["cache_read_write"]+CoT_part, True
    else:
      return "", False
  except:
    return "", False
  
def construct_layout_transformation_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    simplified_transformed_expressions = [re.sub(r'\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}', '', re.sub(r'\^{[a-zA-Z0-9,]*}', '', item)).replace('}','').replace("L","").replace("P","").replace("V","").replace("B","").replace("U","").replace("[","").replace("]","") for item in transformed_expressions]
    # print(f"simplified_original_expressions:{simplified_original_expressions}, simplified_transformed_expressions:{simplified_transformed_expressions}")
    original_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in original_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in original_expressions]
    original_variables=[itemitem for item in original_variables for itemitem in item]
    original_variables_filtered=list(set([item for item in original_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    simplified_original_variable=list(set([item[:item.index('^')] for item in original_variables_filtered]))
    transformed_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in transformed_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in transformed_expressions]
    transformed_variables=[itemitem for item in transformed_variables for itemitem in item]
    transformed_variables_filtered=list(set([item for item in transformed_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    simplified_transformed_variables=list(set([item[:item.index('^')] for item in transformed_variables_filtered]))
    # print(f"simplified_original_variable:{simplified_original_variable}, simplified_transformed_variables:{simplified_transformed_variables}")
    already_find=False
    original_variable, transformed_variable, expr_idx="", "", -1
    for original_var in simplified_original_variable:
      for transformed_var in simplified_transformed_variables:
        for item_idx in range(len(simplified_transformed_expressions)):
          if transformed_var+'='+original_var+';' in simplified_transformed_expressions[item_idx]:
            already_find=True
            original_variable=original_var
            transformed_variable=transformed_var
            expr_idx=item_idx
            break
        if already_find:
          break
      if already_find:
        break
    if original_variable!="" and transformed_variable!="" and expr_idx!=-1:
       CoT_part=f" Under the given IR, the memory layout of variable \'{original_variable}\' can be transformed as the variable \'{transformed_variable}\' by the expression \'{transformed_expressions[expr_idx]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["layout_transformation"]+CoT_part, True
  except:
    return "", False


def construct_set_storage_scope_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    original_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in original_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in original_expressions]
    original_variables=[itemitem for item in original_variables for itemitem in item]
    original_variables_filtered=list(set([item for item in original_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    transformed_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in transformed_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in transformed_expressions]
    transformed_variables=[itemitem for item in transformed_variables for itemitem in item]
    transformed_variables_filtered=list(set([item for item in transformed_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    set_variable=[item for item in transformed_variables_filtered if ',l}' in item or ',s}' in item]
    assert len(set_variable)>0
    set_original_variable=[item for item in original_variables_filtered if item[:item.index('^')+1] in set_variable[0]]
    assert len(set_original_variable)>0
    # print(f"original_variables_filtered:{original_variables_filtered}, transformed_variables_filtered:{transformed_variables_filtered}, set_variable:{set_variable}, set_original_variable:{set_original_variable}")
    if ',l}' in set_variable[0]:
      CoT_part=f" Under the given IR, the intermediate variable \'{set_original_variable[0]}\' can be set in local memory as \'{set_variable[0]}\'."
    elif ',s}' in set_variable[0]:
      CoT_part=f" Under the given IR, the intermediate variable \'{set_original_variable[0]}\' can be set in shared memory as \'{set_variable[0]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["set_storage_scope"]+CoT_part, True
  except:
    return "", False

def construct_set_storage_layout_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    original_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in original_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in original_expressions]
    original_variables=[itemitem for item in original_variables for itemitem in item]
    original_variables_filtered=list(set([item for item in original_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    transformed_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in transformed_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in transformed_expressions]
    transformed_variables=[itemitem for item in transformed_variables for itemitem in item]
    transformed_variables_filtered=list(set([item for item in transformed_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    # print(f"original_variables_filtered:{original_variables_filtered}, transformed_variables_filtered:{transformed_variables_filtered}")
    assert len(original_variables_filtered)==len(transformed_variables_filtered)
    selected_variable=[[original_item, transformed_item] for original_item in original_variables_filtered for transformed_item in transformed_variables_filtered if original_item!=transformed_item and original_item[:original_item.index('^')]==transformed_item[:transformed_item.index('^')]]
    assert len(selected_variable)>0
    # print(f"selected_variable:{selected_variable}")
    CoT_part=f" Under the given IR, the memory layout of the intermediate variable \'{selected_variable[0][0][:selected_variable[0][0].index('^')]}\' can be directly set by updating its subscripts (e.g., from \'{selected_variable[0][0][selected_variable[0][0].index('_')+1:]}\' to \'{selected_variable[0][1][selected_variable[0][1].index('_')+1:]}\'). Note: all occurrences of \'{selected_variable[0][0][:selected_variable[0][0].index('^')]}\' in the current IR are updated consistently."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["set_storage_layout"]+CoT_part, True
  except:
    return "", False

def construct_precompute_indices_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    original_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in original_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in original_expressions]
    original_variables=[itemitem for item in original_variables for itemitem in item]
    original_variables_filtered=list(set([item for item in original_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    simplified_original_variables=[item[:item.index('^')] for item in original_variables_filtered]
    transformed_variables=[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*}',item) for item in transformed_expressions]+[re.findall(r'[a-zA-Z]+\^{[a-zA-Z0-9,]*}(?!\_{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})',item) for item in transformed_expressions]
    transformed_variables=[itemitem for item in transformed_variables for itemitem in item]
    transformed_variables_filtered=list(set([item for item in transformed_variables if "L^" not in item and "P^" not in item and "V^" not in item and "B^" not in item and "U^" not in item]))
    simplified_transformed_variables=[item[:item.index('^')] for item in transformed_variables_filtered]
    # print(f"original_variables_filtered:{original_variables_filtered}, transformed_variables_filtered:{transformed_variables_filtered}")
    assert len(original_variables_filtered)+1==len(transformed_variables_filtered)
    selected_original_variable=[[original_item, transformed_item] for original_item in original_variables_filtered for transformed_item in transformed_variables_filtered if original_item!=transformed_item and original_item[:original_item.index('^')]==transformed_item[:transformed_item.index('^')]]
    # print(f"selected_original_variable:{selected_original_variable}")
    assert len(selected_original_variable)>0
    prefix=os.path.commonprefix(selected_original_variable[0])
    profix=os.path.commonprefix([selected_original_variable[0][0][::-1], selected_original_variable[0][1][::-1]])
    diff_result=[selected_original_variable[0][0][len(prefix): -len(profix)], selected_original_variable[0][1][len(prefix): -len(profix)]]
    # print(f"diff_result:{diff_result}")
    assert len(diff_result[0])>0 and len(diff_result[1])>0
    simplified_precomputed_variable=[transformed_item for transformed_item in simplified_transformed_variables if transformed_item not in simplified_original_variables]
    precomputed_variable=[item for item in transformed_variables_filtered if simplified_precomputed_variable[0]+'^'==item[:item.index('^')+1]]
    # print(f"simplified_precomputed_variable:{simplified_precomputed_variable}, precomputed_variable:{precomputed_variable}")
    assert len(precomputed_variable)>0
    selected_expression=[item for item in transformed_expressions if precomputed_variable[0]+'=' in item]
    # eq_indices=selected_expression[0][selected_expression[0].index(precomputed_variable[0])+1:]
    # indices=eq_indices[eq_indices.index("=")+1:].replace(";","").replace("]","")
    assert len(selected_expression)>0
    CoT_part=f" Under the given IR, the indices \'{diff_result[0]}\' in the expression \'{original_expressions[0]}\' can be precomputed as the variable \'{diff_result[1]}\' by the expression \'{selected_expression[0]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["precompute_indices"]+CoT_part, True
  except:
    return "", False

def construct_factorization_CoT(original_row_equations, transformed_row_equations, strategy_name):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  strategy_name_CoT=strategy_name.replace("_"," ")
  try:
    assert len(original_expressions)==1 and len(transformed_expressions)==1
    CoT_part=f" Under the given IR, the equation in the expression \'{original_expressions[0]}\' can use the {strategy_name_CoT} strategy to transform as the expression \'{transformed_expressions[0]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict[strategy_name]+CoT_part, True
  except:
    return "", False

def construct_partially_equivalent_then_correct_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  transformed_expressions=[item for item in transformed_row_equations if item in transformed_expressions]
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==2
    assert len(transformed_expressions)==6
    correct_part=''.join(transformed_expressions[4:])
    CoT_part=f" Under the given IR, for the expressions \'{original_expressions[0]}\' and \'{original_expressions[1]}\', the inputs can be concatenated by \'{transformed_expressions[0]}\', the similar operations are executed by \'{transformed_expressions[1]}\' and \'{transformed_expressions[2]}\', then the output can be split by \'{transformed_expressions[3]}\', and finally the results are corrected by \'{correct_part}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["partially_equivalent_then_correct"]+CoT_part, True
  except:
    return "", False

def construct_exponential_split_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==len(transformed_expressions)
    assert len(original_expressions)==1
    sm=difflib.SequenceMatcher(None, original_expressions[0], transformed_expressions[0])
    diff_result=[transformed_expressions[0][j1:j2] for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag!="euqal" and original_expressions[0][i1:i2]!=transformed_expressions[0][j1:j2]]
    split_var=re.findall(r'exp\((.*?)\)', diff_result[0])
    CoT_part=f" Under the given IR, for the expression \'{original_expressions[0]}\', an exponential term \'exp({split_var[0]})\' can be split from the original equation so that the expression \'{transformed_expressions[0]}\' is obtained."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["exponential_split"]+CoT_part, True
  except:
    return "", False

def construct_multiplicative_split_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==len(transformed_expressions)
    assert len(original_expressions)==1
    sm=difflib.SequenceMatcher(None, original_expressions[0], transformed_expressions[0])
    diff_result=[transformed_expressions[0][j1:j2] for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag!="euqal" and original_expressions[0][i1:i2]!=transformed_expressions[0][j1:j2]]
    # print(f"diff_result:{diff_result}")
    split_var=[item[item.index('*'):-1] if ')'==item[-1] else item[item.index('*'):] for item in diff_result if '*' in item and '/' in item]
    # print(f"diff_result:{diff_result}, split_var:{split_var}")
    CoT_part=f" Under the given IR, for the expression \'{original_expressions[0]}\', a multiplicative term \'{split_var[0]}\' can be split from the original equation so that the expression \'{transformed_expressions[0]}\' is obtained."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["exponential_split"]+CoT_part, True
  except:
    return "", False

def construct_additive_split_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==len(transformed_expressions)
    assert len(original_expressions)==1
    sm=difflib.SequenceMatcher(None, original_expressions[0], transformed_expressions[0])
    diff_result=[transformed_expressions[0][j1:j2] for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag!="euqal" and original_expressions[0][i1:i2]!=transformed_expressions[0][j1:j2]]
    # print(f"diff_result:{diff_result}")
    split_var=[item[item.index('-'):-1] if '}'!=item[-1] else item[item.index('-'):] for item in diff_result if '-' in item]
    # print(f"diff_result:{diff_result}, split_var:{split_var}")
    CoT_part=f" Under the given IR, for the expression \'{original_expressions[0]}\', an additive term \'{split_var[0]}\' can be split from the original equation so that the expression \'{transformed_expressions[0]}\' is obtained."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["additive_split"]+CoT_part, True
  except:
    return "", False

def construct_normal_loop_max_to_prefix_max_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  transformed_expressions=[item for item in transformed_row_equations if item in transformed_expressions]
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==1
    assert len(transformed_expressions)==2
    CoT_part=f" Under the given IR, for the expression \'{original_expressions[0]}\' with the maximum operation, online streaming method can be used via initializing the output by \'{transformed_expressions[0]}\' and then updating the next step based on the previous step by \'{transformed_expressions[0]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["normal_loop_max_to_prefix_max"]+CoT_part, True
  except:
    return "", False

def construct_normal_loop_summation_on_exp_to_prefix_summation_on_exp_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  original_expressions=[item for item in original_row_equations if item in original_expressions]
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  transformed_expressions=[item for item in transformed_row_equations if item in transformed_expressions]
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert "max" in original_expressions[0] and "exp" in original_expressions[1]
    CoT_part=f" Under the given IR, for the expression \'{original_expressions[1]}\' with the summation operation on the exponential term and its previous expression \'{original_expressions[0]}\' with the maximum operation, online streaming method can be used via updating the next step based on the previous step by \'{transformed_expressions[0]}\' and writing the output by \'{transformed_expressions[1]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["normal_loop_summation_on_exp_to_prefix_summation_on_exp"]+CoT_part, True
  except:
    return "", False

def construct_online_softmax_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  original_expressions=[item for item in original_row_equations if item in original_expressions]
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  transformed_expressions=[item for item in transformed_row_equations if item in transformed_expressions]
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==3 and "max" in original_expressions[0] and "exp" in original_expressions[1] and "exp" in original_expressions[2]
    softmax=''.join(original_expressions)
    CoT_part=f" Under the given IR, for the softmax expressions \'{softmax}\', the online softmax can be used by \'{transformed_expressions[0]}\' and the output can be written by \'{transformed_expressions[1]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["normal_loop_summation_on_exp_to_prefix_summation_on_exp"]+CoT_part, True
  except:
    return "", False

def construct_flashattention_wo_tiling_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  original_expressions=[item for item in original_row_equations if item in original_expressions]
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  transformed_expressions=[item for item in transformed_row_equations if item in transformed_expressions]
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==4 and "max" in original_expressions[0] and "exp" in original_expressions[1] and "exp" in original_expressions[2]
    softmax=''.join(original_expressions[:3])
    CoT_part=f" Under the given IR, for the softmax expressions \'{softmax}\' and the matmul expression \'{original_expressions[3]}\', the flashattention without tiling can be used by \'{transformed_expressions[0]+transformed_expressions[1]}\' and the output can be written by \'{transformed_expressions[2]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["flashattention_wo_tiling"]+CoT_part, True
  except:
    return "", False

def construct_normal_matmul_to_prefix_matmul_based_on_online_softmax_CoT(original_row_equations, transformed_row_equations):
  original_expressions=list(set(original_row_equations)-set(transformed_row_equations))
  original_expressions=[item for item in original_row_equations if item in original_expressions]
  transformed_expressions=list(set(transformed_row_equations)-set(original_row_equations))
  transformed_expressions=[item for item in transformed_row_equations if item in transformed_expressions]
  # print(f"original_expressions:{original_expressions}\n transformed_expressions:{transformed_expressions}")
  try:
    assert len(original_expressions)==3 and "max" in original_expressions[0] and "exp" in original_expressions[0] and "exp" in original_expressions[1]
    online_softmax=''.join(original_expressions[:2])
    CoT_part=f" Under the given IR, for the online softmax expressions \'{online_softmax}\' and the matmul expression \'{original_expressions[2]}\', the online method can be used by \'{transformed_expressions[0]+transformed_expressions[1]}\' and the output can be written by \'{transformed_expressions[2]}\'."
    # print(f"CoT_part:{CoT_part}")
    return strategy_dict["normal_matmul_to_prefix_matmul_based_on_online_softmax"]+CoT_part, True
  except:
    return "", False

def construct_CoT_prompt(original_IR, transformed_IR, strategy_name):
  # print(f'original_IR:{original_IR}\n transformed_IR:{transformed_IR}')
  original_row_equations, original_loops, original_equations, original_eq_outputs, original_eq_inputs, original_simplified_eqs, original_simplified_eq_outputs, original_simplified_eq_inputs = split_IR_to_equations(original_IR)
  # print(f"original_row_equations:{original_row_equations}\n original_loops:{original_loops}\n original_equations:{original_equations}\n original_eq_outputs:{original_eq_outputs}\n original_eq_inputs:{original_eq_inputs}\n original_simplified_eqs:{original_simplified_eqs}\n original_simplified_eq_outputs:{original_simplified_eq_outputs}\n original_simplified_eq_inputs:{original_simplified_eq_inputs}")
  transformed_row_equations, transformed_loops= split_transformed_IR_to_equations(transformed_IR)
  CoT_prompt=""
  correct_strategy=True
  if strategy_name=="operator_fusion":
    CoT_prompt, correct_strategy=construct_operator_fusion_CoT(original_row_equations, transformed_row_equations, transformed_loops)
  elif strategy_name=="operator_fission":
    CoT_prompt, correct_strategy=construct_operator_fission_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="compute_inline":
    CoT_prompt, correct_strategy=construct_compute_inline_CoT(original_row_equations, transformed_row_equations, transformed_loops)
  elif strategy_name=="expression_splitting":
    CoT_prompt, correct_strategy=construct_expression_splitting_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="tensor_concat_to_fuse_operators":
    CoT_prompt, correct_strategy=construct_tensor_concat_to_fuse_operators_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="tensor_split_to_decouple_operators":
    CoT_prompt, correct_strategy=construct_tensor_split_to_decouple_operators_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="common_subexpression_elimination":
    CoT_prompt, correct_strategy=construct_common_subexpression_elimination_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="expression_reorder":
    CoT_prompt, correct_strategy=construct_expression_reorder_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="loop_reorder":
    CoT_prompt, correct_strategy=construct_loop_reorder_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops)
  elif strategy_name=="loop_tiling":
    CoT_prompt, correct_strategy=construct_loop_tiling_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops)
  elif strategy_name=="loop_split":
    CoT_prompt, correct_strategy=construct_loop_split_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops)
  elif strategy_name=="loop_fusion":
    CoT_prompt, correct_strategy=construct_loop_fusion_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops)
  elif strategy_name=="loop_unrolling":
    CoT_prompt, correct_strategy=construct_loop_unrolling_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops)
  elif strategy_name=="loop_parallelization":
    CoT_prompt, correct_strategy=construct_loop_parallelization_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops)
  elif strategy_name=="loop_vectorization":
    CoT_prompt, correct_strategy=construct_loop_vectorization_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops)
  elif strategy_name=="loop_binding":
    CoT_prompt, correct_strategy=construct_loop_binding_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops)
  elif strategy_name=="reduction_factorization":
    CoT_prompt, correct_strategy=construct_reduction_factorization_CoT(original_row_equations, transformed_row_equations, original_loops, transformed_loops)
  elif strategy_name=="cache_read_write":
    CoT_prompt, correct_strategy=construct_cache_read_write_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="layout_transformation":
    CoT_prompt, correct_strategy=construct_layout_transformation_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="set_storage_scope":
    CoT_prompt, correct_strategy=construct_set_storage_scope_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="set_storage_layout":
    CoT_prompt, correct_strategy=construct_set_storage_layout_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="precompute_indices":
    CoT_prompt, correct_strategy=construct_precompute_indices_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name in ["factorization","expand_factorization","cancellation","expand_cancellation","apart","together","powsimp","expand_powsimp","logsimp","expand_log","collect","expand_collect"]:
    CoT_prompt, correct_strategy=construct_factorization_CoT(original_row_equations, transformed_row_equations, strategy_name)
  elif strategy_name=="partially_equivalent_then_correct":
    CoT_prompt, correct_strategy=construct_partially_equivalent_then_correct_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="exponential_split":
    CoT_prompt, correct_strategy=construct_exponential_split_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="multiplicative_split":
    CoT_prompt, correct_strategy=construct_multiplicative_split_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="additive_split":
    CoT_prompt, correct_strategy=construct_additive_split_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="normal_loop_max_to_prefix_max":
    CoT_prompt, correct_strategy=construct_normal_loop_max_to_prefix_max_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="normal_loop_summation_on_exp_to_prefix_summation_on_exp":
    CoT_prompt, correct_strategy=construct_normal_loop_summation_on_exp_to_prefix_summation_on_exp_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="online_softmax":
    CoT_prompt, correct_strategy=construct_online_softmax_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="flashattention_wo_tiling":
    CoT_prompt, correct_strategy=construct_flashattention_wo_tiling_CoT(original_row_equations, transformed_row_equations)
  elif strategy_name=="normal_matmul_to_prefix_matmul_based_on_online_softmax":
    CoT_prompt, correct_strategy=construct_normal_matmul_to_prefix_matmul_based_on_online_softmax_CoT(original_row_equations, transformed_row_equations)
  return CoT_prompt, correct_strategy

def delete_unvalid_data(dataset_with_CoT, delete_data):
  new_dataset=[]
  num_multiple=0
  num_single=0
  multi_strategy_dict={}
  single_strategy_dict={}
  for data_idx, data in tqdm.tqdm(enumerate(dataset_with_CoT)):
    label=ast.literal_eval(data['label'])
    original_IR=data['original_IR']
    if isinstance(label, list):
      CoT_list=ast.literal_eval(data["CoT"])
      new_label=[]
      new_CoT_list=[]
      for label_idx, label_item in enumerate(label):
        if [data_idx, "multiple", label_idx] not in delete_data:
          new_label.append(label_item)
          new_CoT_list.append(CoT_list[label_idx])
      if len(new_label)>1:
        tmp_transformed_set=set([new_label_item['transformed_IR'] for new_label_item in new_label])
        add_multiple_data=False
        if original_IR in multi_strategy_dict:
          if tmp_transformed_set in multi_strategy_dict[original_IR]:
            add_multiple_data=False
          else:
            multi_strategy_dict[original_IR].append(tmp_transformed_set)
            add_multiple_data=True
        else:
          multi_strategy_dict[original_IR]=[tmp_transformed_set]
          add_multiple_data=True
        if add_multiple_data and "" not in new_CoT_list and len(new_label)==len(new_CoT_list):
          TIR_label=ast.literal_eval(str(data['TIR_label']))
          new_TIR_label=[]
          for idx, new_label_item in enumerate(new_label):
            new_TIR_label.append(TIR_label[new_label_item['idx']])
            new_label[idx]['idx']=idx
            new_TIR_label[idx]['idx']=idx
          if len(new_TIR_label)==len(new_label):
            data['prompt']=data['prompt'].replace(f"at least {len(label)}", f"at least {len(new_label)}")
            data["label"]=str(new_label)
            data["TIR_label"]=str(new_TIR_label)
            data["CoT"]=str(new_CoT_list)
            new_dataset.append(data)
            num_multiple+=1
    else:
      if [data_idx, "single"] not in delete_data:
        new_dataset.append(data)
        num_single+=1
        if original_IR in single_strategy_dict:
          if transformed_IR in single_strategy_dict[original_IR]:
            single_strategy_dict[original_IR].append(transformed_IR)
        else:
          single_strategy_dict[original_IR]=[transformed_IR]
  print(f"len(new_dataset):{len(new_dataset)}, num_single:{num_single}, num_multiple:{num_multiple}")
  return new_dataset

def check_strategy_statistic(train_dataset):
  #check the statistic
  labels=[item['label'] for item in train_dataset]
  print(f'len(labels):{len(labels)}')
  #obtain the statistic of strategy number
  strategy_dict={}
  single_strategy_dict={}
  for label in tqdm.tqdm(labels):
    label=ast.literal_eval(label)
    if isinstance(label, list):
      for label_item in label:
        applied_strategy_value=label_item["applied_strategy"]
        if applied_strategy_value in strategy_dict:
          strategy_dict[applied_strategy_value]+=1
        else:
          strategy_dict[applied_strategy_value]=1
    else:
      applied_strategy_value=label["applied_strategy"]
      if applied_strategy_value in strategy_dict:
        strategy_dict[applied_strategy_value]+=1
      else:
        strategy_dict[applied_strategy_value]=1
      if applied_strategy_value in single_strategy_dict:
        single_strategy_dict[applied_strategy_value]+=1
      else:
        single_strategy_dict[applied_strategy_value]=1
  print(f"strategy_dict:{strategy_dict}, sum:{sum(strategy_dict.values())}")
  print(f"single_strategy_dict:{single_strategy_dict}, sum:{sum(single_strategy_dict.values())}")

def construct_label_with_CoT(dataset):
  new_dataset=[]
  num_multiple=0
  num_single=0
  for data_idx, data in tqdm.tqdm(enumerate(dataset)):
    label=ast.literal_eval(data['label'])
    if isinstance(label, list):
      CoT_list=ast.literal_eval(data["CoT"])
      if len(label)==len(CoT_list):
        label_with_CoT=f'<think>These {len(label)} transformed IRs can be individually analyzed as follows: '
        num_CoT=0
        for label_idx, label_item in enumerate(label):
          if CoT_list[label_idx]!="":
            label_with_CoT+=f'{label_idx}. '+CoT_list[label_idx]+'\n'
            num_CoT+=1
        if num_CoT==len(label):
          label_with_CoT+='<\\think><answer>'+data['label']+'<\\answer>'
          data["label_with_CoT"]=label_with_CoT
          new_dataset.append(data)
          num_multiple+=1
    else:
      if data["CoT"]!="":
        label_with_CoT='<think>'+data["CoT"]+'<\\think><answer>'+data['label']+'<\\answer>'
        data["label_with_CoT"]=label_with_CoT
        num_single+=1
        new_dataset.append(data)
  print(f"len(new_dataset):{len(new_dataset)}, num_single:{num_single}, num_multiple:{num_multiple}")
  return new_dataset


if __name__ == '__main__':
  with open("../nfs_folder/data_entries/multi_IRs_train_dataset_filtered2.jsonl", "r") as f:
    train_dataset=[json.loads(line) for line in tqdm.tqdm(f)]
  CoT_database={}
  delete_data=[]
  dataset_with_CoT=[]
  for data_idx, data in tqdm.tqdm(enumerate(train_dataset)):
    label=ast.literal_eval(data['label'])
    original_IR=data['original_IR']
    if original_IR not in CoT_database:
      CoT_database[original_IR]={}
    if isinstance(label, list):
      CoT_list=[]
      for label_idx, label_item in enumerate(label):
        strategy_name=label_item["applied_strategy"]
        transformed_IR=label_item["transformed_IR"]
        if transformed_IR not in CoT_database[original_IR]:
          CoT_prompt, correct_strategy=construct_CoT_prompt(original_IR, transformed_IR, strategy_name)
          if not correct_strategy:
            delete_data.append([data_idx, "multiple", label_idx])
          CoT_database[original_IR][transformed_IR+strategy_name]=CoT_prompt
        else:
          CoT_prompt=CoT_database[original_IR][transformed_IR+strategy_name]
        CoT_list.append(CoT_prompt)
      data["CoT"]=str(CoT_list)
      dataset_with_CoT.append(data)
    else:
      strategy_name=label["applied_strategy"]
      transformed_IR=label["transformed_IR"]
      if transformed_IR not in CoT_database[original_IR]:
        CoT_prompt, correct_strategy=construct_CoT_prompt(original_IR, transformed_IR, strategy_name)
        if not correct_strategy:
          delete_data.append([data_idx, "single"])
        CoT_database[original_IR][transformed_IR+strategy_name]=CoT_prompt
      else:
        CoT_prompt=CoT_database[original_IR][transformed_IR+strategy_name]
      data["CoT"]=CoT_prompt
      dataset_with_CoT.append(data)
  print(f'len(delete_data):{len(delete_data)}')
  single_delete=[item for item in delete_data if item[1]=="single"]
  print(f'len(single_delete):{len(single_delete)}')
  print(f"len(dataset_with_CoT):{len(dataset_with_CoT)}")
  new_dataset=delete_unvalid_data(dataset_with_CoT, delete_data)   
  # check_strategy_statistic(new_dataset)
  new_dataset_with_CoT=construct_label_with_CoT(new_dataset)
  for new_data in tqdm.tqdm(new_dataset_with_CoT):
    with open("../nfs_folder/data_entries/multi_IRs_train_dataset_filtered_with_CoT.jsonl", "a") as f:
      f.write(json.dumps(new_data, ensure_ascii=False) + "\n")
  print(f'len(new_dataset_with_CoT):{len(new_dataset_with_CoT)}')

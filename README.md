# StepTP

This repository contains the code used to construct and use the StepTP dataset for learning one-step and multi-step tensor program transformations. The current public package includes the main dataset-construction code and a chat-template training dataset. The archival dictionary-style dataset is not included in this release and will be released after paper acceptance.

## Environment

The code was developed and tested with the following environment:

- Ubuntu 22.04.4
- CUDA 12.4
- LLVM 17.0.6
- Python 3.10.20
- PyTorch 2.6.0+cu124
- TVM 0.21.dev0
- Other Python packages: `tqdm`, `transformers`, `sympy`, `ms-swift`

## Repository Structure

- `pytorch_model_to_IR.py`: traces PyTorch programs with `torch.fx` and converts them to LEIR.
- `ops/`: utilities for parsing PyTorch operations and emitting LEIR expressions.
- `strategies/`: implementation of the transformation strategies.
- `apply_strategy_to_IR.py`: applies a selected strategy to an LEIR and generates one-step transformed LEIRs.
- `construct_CoT_dataset.py`: constructs reasoning traces for transformed LEIRs.
- `transform_IR_to_TIR.py`: converts LEIRs to TVM TIR and runs strategy-level checks.
- `TIR/`: LEIR-to-TIR lowering, TVM build helpers, and verification utilities.
- `eval_verification.py`, `eval_verification_main.py`, `eval_verification_update.py`: verification and evaluation entry points.
- `filtering.py`, `handle_dataset_v2.py`, `handle_dataset_multi_IRs.py`, `further_handle_dataset.py`: dataset post-processing and filtering utilities.
- `construct_TIR_dataset.py`, `obtain_CUDA.py`: optional scripts for constructing TIR/CUDA-side training data.
- `swift_Qwen.py`: Qwen fine-tuning with MS-Swift.
- `swift_inference.py`: single-step transformation inference.
- `swift_*_optimization.py`: multi-step optimization search variants, including chain, DFS, BFS, BS, GS, and MCTS.
- `model_codes/`: PyTorch model/operator programs used as source workloads. The model files are not expanded here because they are workload definitions rather than the core dataset pipeline.
- `dataset.zip`: public chat-template dataset for training.

## Pipeline Mapping

The paper describes four stages. The corresponding code is:

1. PyTorch-to-LEIR Translator
   - Main entry: `pytorch_model_to_IR.py`
   - Supporting code: `ops/`, `model_codes/`, `level1_model_name.json`, `level2_model_name.json`

2. One-step Strategy-driven Transformation
   - Main entry: `apply_strategy_to_IR.py`
   - Strategy implementations: `strategies/strategy.py`, `strategies/utils.py`
   - Reasoning trace construction: `construct_CoT_dataset.py`
   - Strategy list: `strategy_names.json`

3. LEIR-to-TIR Translator
   - Main entry: `transform_IR_to_TIR.py`
   - Supporting code: `TIR/IR_to_TIR.py`, `TIR/utils.py`

4. Verification and Filtering
   - Verification: `TIR/IR_verfication.py`, `eval_verification.py`, `eval_verification_main.py`, `eval_verification_update.py`
   - Filtering and dataset processing: `filtering.py`, `handle_dataset_v2.py`, `handle_dataset_multi_IRs.py`, `further_handle_dataset.py`

## Dataset

The current public dataset archive contains the chat-template JSONL file used for fine-tuning:

```bash
unzip dataset.zip -d .
ls dataset/
# train_dataset_with_CoT_in_chattemplate.jsonl
```

This file can be passed directly to `swift_Qwen.py`. The archival dictionary dataset and the full intermediate construction artifacts are not included in this release. They will be released after the paper is accepted.

## Training

After extracting `dataset.zip`, fine-tune a Qwen model with MS-Swift:

```bash
python swift_Qwen.py \
  --train_dataset dataset/train_dataset_with_CoT_in_chattemplate.jsonl \
  --model_id_or_path /path/to/Qwen3-8B \
  --output_dir /path/to/output/sft_8B_CoT_filtered \
  --cuda_visible_devices 0,1 \
  --batch_size 1 \
  --epochs 1 \
  --gradient_accumulation_steps 16 \
  --max_length 10240
```

The default script values follow the authors' internal directory layout. When reproducing the training outside that environment, set `--train_dataset`, `--model_id_or_path`, and `--output_dir` explicitly.

## Inference and Evaluation

The current public release does not include the evaluation datasets or store-dictionary files used in the paper experiments. Users can still create their own evaluation prompts and generate local store-dictionary files following the verification step below. The paper evaluation files will be released after acceptance.

For single-step transformation inference, use:

```bash
python swift_inference.py --cuda 0 --batch_size 16
```

For multi-step optimization, use one of the search-based scripts:

```bash
python swift_chain_based_optimization.py
python swift_DFS_based_optimization.py
python swift_BFS_based_optimization.py
python swift_BS_based_optimization.py
python swift_GS_based_optimization.py
python swift_MCTS_based_optimization.py
```

These scripts currently contain author-side default paths for evaluation inputs, checkpoints, and output folders. Replace these paths with your own evaluation prompts and generated store-dictionary files for local experiments. The paper evaluation files will be provided after acceptance.

To verify generated transformed LEIRs with TVM:

First generate the corresponding store-dictionary file. This file is collected by `pytorch_model_to_IR.py` during the PyTorch-to-LEIR translation stage:

```bash
python pytorch_model_to_IR.py \
  --level level1 \
  --model_index 0 \
  --dtype float64 \
  --store_dict_dir store_dict
```

This writes a gzip-compressed pickle file such as `store_dict/<MODEL_NAME>_original_<MODEL_INDEX>.pkl.gz`.

```bash
python eval_verification.py \
  --idx 0 \
  --transformed_IR "<TRANSFORMED_LEIR>" \
  --info store_dict/<MODEL_NAME>_original_<MODEL_INDEX>.pkl.gz \
  --num_IRs 1 \
  --cuda 0 \
  --error_record_dir /path/to/error_record.jsonl
```

The verification scripts require the corresponding store-dictionary files. These files can be generated for local workloads with `pytorch_model_to_IR.py`; the full archival set of store dictionaries used by the paper is not included in the current public release.

## Notes for Reproducibility

- The included `dataset.zip` is sufficient for training with `swift_Qwen.py`.
- End-to-end reconstruction of the full dataset requires the archival dictionary dataset and intermediate artifacts, which will be released after acceptance.
- Single-step and multi-step inference/evaluation can be run with user-created evaluation prompts and locally generated store-dictionary files; the paper evaluation files will be released after acceptance.
- The TVM verification scripts require CUDA-enabled and LLVM-enabled TVM and access to the store dictionaries referenced by each data entry.
- Some inference and optimization scripts use hard-coded author-side paths; update these paths before running them in a new environment.

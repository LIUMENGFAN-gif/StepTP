# import some libraries
import os
import gzip
import json
from swift.llm import get_model_tokenizer, get_template, EncodePreprocessor, load_dataset
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial
import torch
import argparse

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model transformations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--save_total_limit', type=int, default=4)
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=10240)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--model_id_or_path', type=str, default='../nfs_folder/LLM/Qwen3-8B')
    parser.add_argument('--output_dir', type=str, default='../nfs_folder/training/sft_8B_CoT_filtered')
    parser.add_argument('--train_dataset', type=str, default='dataset/train_dataset_with_CoT_in_chattemplate.jsonl')
    parser.add_argument('--cuda_visible_devices', type=str, default='1,2')
    parser.add_argument('--num_proc', type=int, default=4)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    torch.cuda.synchronize()
    logger = get_logger()
    seed_everything(42)
    system = 'You are a helpful assistant.'
    output_dir=args.output_dir
    max_length=args.max_length
    model_id_or_path=args.model_id_or_path
    # output_dir="../nfs_folder/training/sft_32B_CoT_filtered"
    # max_length=args.max_length
    # model_id_or_path='../nfs_folder/LLM/Qwen3-32B'
    # output_dir="../nfs_folder/training/sft_1_7B_CoT_filtered"
    # max_length=args.max_length
    # model_id_or_path='../nfs_folder/LLM/Qwen3-1.7B'
    # output_dir="../nfs_folder/training/sft_14B_CoT_filtered"
    # max_length=args.max_length
    # model_id_or_path='../nfs_folder/LLM/Qwen3-14B'

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        weight_decay=args.wd,
        lr_scheduler_type='cosine',
        warmup_ratio=args.warmup_ratio,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        metric_for_best_model='loss',
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        data_seed=42,
    )
    model, tokenizer = get_model_tokenizer(
        model_id_or_path=model_id_or_path)
    logger.info(f'model_info: {model.model_info}')
    template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
    template.set_mode('train')
    target_modules = find_all_linears(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type='CAUSAL_LM',
        target_modules=target_modules,
    )
    model = Swift.prepare_model(model, lora_config)
    logger.info(f'lora_config: {lora_config}')
    # Print model structure and trainable parameters.
    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')

    train_dataset, _ = load_dataset(args.train_dataset, num_proc=args.num_proc, seed=42)
    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=args.num_proc)
    print(f'len(train_dataset):{len(train_dataset)}')


    # Print a sample
    template.print_inputs(train_dataset[0])
    # Get the trainer and start the training.
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        template=template,
    )
    trainer.train()

    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')

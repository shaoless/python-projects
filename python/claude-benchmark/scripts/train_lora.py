#!/usr/bin/env python
"""Fine-tune a pre-trained language model with LoRA (Low-Rank Adaptation).

This is the practical, industry-standard approach:
    - Load a pre-trained model (e.g., SmolLM2-360M, Qwen2.5-0.5B)
    - Freeze the base model
    - Attach small trainable "adapter" matrices to attention layers
    - Train only the adapters (~1% of parameters)

Usage:
    # Basic LoRA fine-tuning:
    python scripts/train_lora.py --model HuggingFaceTB/SmolLM2-360M --max_steps 500

    # With 4-bit quantization (QLoRA, saves VRAM):
    python scripts/train_lora.py --use_4bit --model Qwen/Qwen2.5-0.5B

Note:
    bitsandbytes is required for 4-bit quantization — may not work on Windows.
    On Windows, omit --use_4bit to run in full precision LoRA.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))


def format_instruction(example: dict) -> str:
    """Format an Alpaca-style instruction into a text prompt."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        text = (
            f"Below is an instruction that describes a task, paired with an input. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        )
    else:
        text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune a language model")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M",
                        help="Base model from HuggingFace")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (QLoRA — requires bitsandbytes)")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora-adapter")
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned",
                        help="Instruction dataset to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size (for quick tests)")
    args = parser.parse_args()

    print("=" * 60)
    print("LoRA Fine-Tuning")
    print(f"  Base model: {args.model}")
    print(f"  4-bit QLoRA: {args.use_4bit}")
    print(f"  LoRA rank: {args.lora_r} | alpha: {args.lora_alpha}")
    print(f"  Steps: {args.max_steps} | LR: {args.lr}")
    print("=" * 60)

    # --- Load model with PEFT ---
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    if args.use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load dataset ---
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset)
    train_data = dataset["train"]
    if args.max_samples:
        train_data = train_data.select(range(args.max_samples))

    def tokenize_fn(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            inst = examples["instruction"][i]
            inp = examples.get("input", [""] * len(examples["instruction"]))[i]
            out = examples["output"][i]
            texts.append(format_instruction({"instruction": inst, "input": inp, "output": out}))

        result = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    print("Tokenizing...")
    train_data = train_data.map(tokenize_fn, batched=True, remove_columns=train_data.column_names)
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to=None,                  # No wandb, no tensorboard (keep it simple)
        dataloader_num_workers=2,
        gradient_checkpointing=True,
    )

    # --- Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    # --- Save adapter ---
    adapter_path = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to: {adapter_path}")
    print("Done!")


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local AI model training project for learning how large language models are trained. It has two main workflows:

1. **Train Mini-GPT from scratch** — a 25M-parameter GPT-2 style Transformer in pure PyTorch (educational)
2. **LoRA fine-tuning** — industrial-standard fine-tuning of pre-trained models using PEFT (practical)

Target hardware: RTX 4060 8GB VRAM / 16GB RAM / Windows 11. All model sizes and batch sizes are chosen to fit within 8GB VRAM.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download training data
python data/download.py tinystories    # Simple stories (recommended for mini-GPT)
python data/download.py wikitext       # WikiText-2
python data/download.py alpaca         # Instruction data for LoRA

# Train Mini-GPT (quick test: 100 steps, ~2 min)
python scripts/train_gpt.py --max_steps 100 --eval_interval 50

# Train Mini-GPT (full: 10000 steps, ~30-60 min on RTX 4060)
python scripts/train_gpt.py --max_steps 10000 --batch_size 8 --grad_accum 4

# Resume training from checkpoint
python scripts/train_gpt.py --resume checkpoints/step_005000.pt

# Generate text from trained Mini-GPT
python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"

# LoRA fine-tuning (without 4-bit, works on Windows)
python scripts/train_lora.py --model HuggingFaceTB/SmolLM2-360M --max_steps 500

# LoRA with QLoRA 4-bit (Linux only, bitsandbytes required)
python scripts/train_lora.py --use_4bit --model Qwen/Qwen2.5-0.5B --max_steps 500

# Generate from LoRA adapter
python scripts/generate.py --lora-adapter checkpoints/lora-adapter/adapter --prompt "Explain:"

# View training curves
tensorboard --logdir logs
```

## Architecture

**`src/model.py`** — Pure PyTorch GPT-2 style Transformer Decoder:
- `CausalSelfAttention` — Multi-head attention with causal masking
- `MLP` — GELU feed-forward with 4x expansion
- `TransformerBlock` — Pre-norm attention + MLP with residual connections
- `MiniGPT` — Full model: token/position embeddings → N blocks → LM head. Includes `generate()` with top-k/top-p sampling and `configure_optimizers()` for AdamW with selective weight decay.

**`src/trainer.py`** — Generic training loop:
- Automatic mixed precision (AMP) via `torch.autocast`
- Gradient accumulation for larger effective batch sizes
- Cosine LR schedule with linear warmup
- Periodic eval, checkpointing with rotation (keeps last N checkpoints)
- TensorBoard logging

**`src/config.py`** — Three config dataclasses: `MiniGPTConfig` (model architecture), `TrainingConfig` (optimization), `LoRAConfig` (fine-tuning).

**`src/dataset.py`** — Two dataset types: `TextDataset` (in-memory, for smaller datasets) and `StreamingDataset` (on-the-fly, for datasets larger than RAM).

**`src/tokenizer.py`** — Uses `tiktoken` for pretrained tokenizers (GPT-2) and `tokenizers` (HuggingFace) for custom BPE training.

**Entry scripts** (`scripts/`):
- `train_gpt.py` — Wires MiniGPT + data + trainer together. CLI flags override all hyperparameters.
- `train_lora.py` — Uses HuggingFace `transformers` + `peft` + `datasets`. Formats Alpaca instructions, tokenizes, trains with `Trainer`.
- `generate.py` — Supports both MiniGPT checkpoints and LoRA adapters. Top-k + top-p sampling.

## Key Constraints

- **8GB VRAM**: Mini-GPT at 25M params with batch=8, block_size=512, bfloat16 fits comfortably. Larger models need gradient checkpointing or QLoRA.
- **Windows**: `bitsandbytes` does not officially support Windows. Always provide a non-4bit fallback for LoRA training. `train_lora.py` defaults to no quantization.
- **16GB RAM**: `TextDataset` loads the entire tokenized corpus into memory. For datasets >100MB of raw text, use `StreamingDataset` instead.
- **Model checkpoint**: The save format includes model state_dict, optimizer state, scaler state, step count, and val_loss. The model config is expected under the `model_config` key for `generate.py`.

## Tokenizer Notes

- Mini-GPT uses `tiktoken` with "gpt2" encoding (50257 vocab). For training from scratch with a custom tokenizer, use `train_hf_tokenizer()` from `tokenizer.py`.
- LoRA scripts use the tokenizer from the base model (via `AutoTokenizer`).

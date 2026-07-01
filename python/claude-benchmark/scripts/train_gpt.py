#!/usr/bin/env python
"""Train a Mini-GPT model from scratch.

Usage:
    # Quick test (100 steps):
    python scripts/train_gpt.py --max_steps 100 --eval_interval 50

    # Full training (10k steps, ~30-60 min on RTX 4060):
    python scripts/train_gpt.py --max_steps 10000

    # Resume from checkpoint:
    python scripts/train_gpt.py --resume checkpoints/step_005000.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MiniGPTConfig, TrainingConfig
from src.model import MiniGPT
from src.dataset import create_dataloaders
from src.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mini-GPT from scratch")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_data", type=int, default=0,
                        help="Max chars to load (0=all). For quick tests, try 10_000_000 (10MB)")
    args = parser.parse_args()

    # --- Config ---
    model_cfg = MiniGPTConfig(block_size=args.block_size)
    train_cfg = TrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        dtype=args.dtype,
        compile=args.compile,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )

    print("=" * 60)
    print("Mini-GPT Training")
    print(f"  Model: {model_cfg.estimated_params() / 1e6:.1f}M params (est.)")
    print(f"  Steps: {train_cfg.max_steps}")
    print(f"  Batch: {train_cfg.batch_size} × {train_cfg.grad_accum_steps} accum")
    print(f"  LR: {train_cfg.learning_rate}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # --- Data files ---
    # Look for prepared text files (priority: TinyStories > WikiText)
    train_file = None
    val_file = None

    candidates = [
        ("data/tinystories_train.txt", "data/tinystories_validation.txt"),
        ("data/wikitext_train.txt", "data/wikitext_validation.txt"),
    ]

    for t, v in candidates:
        if os.path.exists(t):
            train_file, val_file = t, v
            print(f"Using dataset: {t}")
            break

    if train_file is None:
        print("ERROR: No training data found!")
        print("Run: python data/download.py tinystories")
        print("  or: python data/download.py wikitext")
        sys.exit(1)

    # --- Dataloaders ---
    train_loader, val_loader = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        block_size=model_cfg.block_size,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        max_chars=args.max_data,
    )

    # --- Model ---
    model = MiniGPT(model_cfg)

    # --- Trainer ---
    trainer = Trainer(model, train_cfg, model_config=model_cfg)

    # --- Train ---
    trainer.train(train_loader, val_loader, resume_from=args.resume)

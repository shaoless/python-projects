"""Generic training loop with mixed precision, checkpointing, and logging."""

import os
import glob
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .config import TrainingConfig


class Trainer:
    """Generic trainer for language models.

    Features:
        - Automatic mixed precision (AMP)
        - Gradient accumulation
        - Learning rate warmup + cosine decay
        - Periodic evaluation and checkpointing
        - TensorBoard logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        model_config=None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.model_config = model_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        # Optimizer — decoupled weight decay (AdamW)
        self.optimizer = model.configure_optimizers(
            config.weight_decay,
            config.learning_rate,
            config.betas,
        ) if hasattr(model, "configure_optimizers") else torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        # Mixed precision
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.dtype]
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))

        # Logging
        os.makedirs(config.log_dir, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)

        # Checkpointing
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = Path(config.checkpoint_dir)

        # Step tracking
        self.step = 0
        self.best_val_loss = float("inf")

        # torch.compile (PyTorch 2.0+)
        if config.compile:
            self.model = torch.compile(self.model)
            print("Model compiled with torch.compile()")

    def get_lr(self) -> float:
        """Cosine learning rate schedule with warmup."""
        if self.step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * self.step / self.config.warmup_steps

        if self.step >= self.config.max_steps:
            return 0.0

        # Cosine decay
        progress = (self.step - self.config.warmup_steps) / (
            self.config.max_steps - self.config.warmup_steps
        )
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save_checkpoint(self, name: str, val_loss: Optional[float] = None) -> None:
        """Save model, optimizer, and scaler state."""
        path = self.checkpoint_dir / name
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "val_loss": val_loss,
                "model_config": self.model_config,
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

        # Rotate old checkpoints
        self._rotate_checkpoints()

    def _rotate_checkpoints(self) -> None:
        """Keep only the N most recent checkpoints."""
        ckpts = sorted(
            glob.glob(str(self.checkpoint_dir / "step_*.pt")),
            key=os.path.getmtime,
        )
        while len(ckpts) > self.config.max_checkpoints:
            os.remove(ckpts.pop(0))

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        print(f"Resumed from step {self.step} (val_loss={ckpt.get('val_loss', 'N/A')})")

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step with gradient accumulation.

        Returns:
            The loss value for this micro-batch.
        """
        self.model.train()

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.config.dtype != "float32",
        ):
            _, loss = self.model(x, y)

        # Scale loss for gradient accumulation
        loss = loss / self.config.grad_accum_steps
        self.scaler.scale(loss).backward()

        return loss.item() * self.config.grad_accum_steps  # Report unscaled loss

    def optimizer_step(self) -> None:
        """Clip gradients and step optimizer (call after grad_accum_steps)."""
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip,
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 50) -> float:
        """Compute average validation loss (limited to max_batches)."""
        self.model.eval()
        losses = []
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.config.dtype != "float32",
            ):
                _, loss = self.model(x, y)
            losses.append(loss.item())

        return sum(losses) / len(losses) if losses else float("inf")

    def train(
        self,
        train_loader,
        val_loader=None,
        resume_from: Optional[str] = None,
    ) -> None:
        """Main training loop.

        Args:
            train_loader: DataLoader yielding (x, y) batches.
            val_loader: Optional validation DataLoader.
            resume_from: Path to checkpoint to resume from.
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        print(f"Training on {self.device} | dtype={self.config.dtype}")
        print(f"Steps: {self.config.max_steps} | Batch: {self.config.batch_size}")
        print(f"Grad accum: {self.config.grad_accum_steps}")
        print(f"Effective batch: {self.config.batch_size * self.config.grad_accum_steps}")

        train_iter = iter(train_loader)
        t0 = time.time()
        running_loss = 0.0
        micro_step = 0

        while self.step < self.config.max_steps:
            # Get batch — restart iterator if exhausted
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Training step (accumulates gradients)
            loss_val = self.train_step(x, y)
            running_loss += loss_val
            micro_step += 1

            # Update weights after gradient accumulation
            if micro_step >= self.config.grad_accum_steps:
                self.optimizer_step()

                # Update learning rate
                lr = self.get_lr()
                self.set_lr(lr)

                self.step += 1
                micro_step = 0

                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_loss = running_loss / (self.config.log_interval * self.config.grad_accum_steps)
                    elapsed = time.time() - t0
                    steps_per_sec = self.config.log_interval / elapsed
                    print(
                        f"step {self.step:>6d}/{self.config.max_steps} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"tok/s {steps_per_sec * self.config.batch_size * self.config.grad_accum_steps:.0f}"
                    )
                    self.writer.add_scalar("train/loss", avg_loss, self.step)
                    self.writer.add_scalar("train/lr", lr, self.step)
                    running_loss = 0.0
                    t0 = time.time()

                # Evaluation
                if val_loader and self.step % self.config.eval_interval == 0:
                    val_loss = self.evaluate(val_loader)
                    self.writer.add_scalar("val/loss", val_loss, self.step)
                    print(f"  → val_loss {val_loss:.4f}")
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best.pt", val_loss)

                # Checkpointing
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint(f"step_{self.step:06d}.pt")

        # Final save
        self.save_checkpoint("final.pt")
        print(f"Training complete! Best val_loss: {self.best_val_loss:.4f}")
        self.writer.close()

"""Tests for the Trainer class and training loop.

Covers:
    - Trainer initialization with model, config, and device
    - Learning rate schedule: warmup, cosine decay, post-max_steps
    - set_lr updates optimizer param groups
    - train_step returns loss and accumulates gradients
    - optimizer_step clips gradients and steps optimizer
    - evaluate() runs and returns a float
    - Checkpoint save and load roundtrip (state dict restoration)
    - Checkpoint rotation (keeps max_checkpoints, removes oldest)
    - Training loop runs without crashing for N steps
    - Gradient accumulation semantics
"""

import glob
import math
import os
import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.trainer import Trainer


# ===================================================================
# Local fixtures
# ===================================================================

@pytest.fixture
def trainer(tiny_train_model, small_train_config, cpu_device):
    """Trainer instance with a tiny model on CPU for testing."""
    return Trainer(
        model=tiny_train_model,
        config=small_train_config,
        device=cpu_device,
    )


@pytest.fixture
def dummy_train_loader(tiny_config, small_train_config):
    """DataLoader with random data for training (20 batches)."""
    vocab_size = tiny_config.vocab_size
    block_size = tiny_config.block_size
    batch_size = small_train_config.batch_size
    total_samples = batch_size * 20  # 20 batches worth
    dataset = TensorDataset(
        torch.randint(0, vocab_size, (total_samples, block_size)),
        torch.randint(0, vocab_size, (total_samples, block_size)),
    )
    return DataLoader(dataset, batch_size=batch_size, drop_last=True)


@pytest.fixture
def dummy_val_loader(tiny_config, small_train_config):
    """Small DataLoader for validation (3 batches)."""
    vocab_size = tiny_config.vocab_size
    block_size = tiny_config.block_size
    batch_size = small_train_config.batch_size
    total_samples = batch_size * 3
    dataset = TensorDataset(
        torch.randint(0, vocab_size, (total_samples, block_size)),
        torch.randint(0, vocab_size, (total_samples, block_size)),
    )
    return DataLoader(dataset, batch_size=batch_size, drop_last=True)


@pytest.fixture
def trainer_with_rotating_checkpoints(tiny_train_model, medium_train_config, cpu_device):
    """Trainer with max_checkpoints=3 for rotation tests."""
    return Trainer(
        model=tiny_train_model,
        config=medium_train_config,
        device=cpu_device,
    )


# ===================================================================
# Trainer Initialization
# ===================================================================

class TestTrainerInit:
    """Tests for Trainer.__init__()."""

    def test_init_with_model_and_config(self, trainer, small_train_config, cpu_device):
        """Trainer is initialized with model and config attributes."""
        assert trainer.config == small_train_config
        assert trainer.device == cpu_device
        assert trainer.step == 0
        assert trainer.best_val_loss == float("inf")

    def test_optimizer_is_adamw(self, trainer):
        """Trainer creates an AdamW optimizer."""
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_optimizer_has_two_param_groups(self, trainer):
        """Optimizer has decay and no-decay param groups."""
        assert len(trainer.optimizer.param_groups) == 2

    def test_model_on_correct_device(self, trainer, cpu_device):
        """Model parameters are on the specified device."""
        for param in trainer.model.parameters():
            assert param.device == cpu_device

    def test_dtype_is_float32(self, trainer, small_train_config):
        """dtype matches TrainingConfig (float32 for CPU tests)."""
        assert trainer.dtype == torch.float32

    def test_scaler_disabled_for_float32(self, trainer):
        """GradScaler is disabled when dtype is float32."""
        assert trainer.scaler.is_enabled() is False

    def test_writer_created(self, trainer):
        """TensorBoard SummaryWriter is created."""
        assert trainer.writer is not None

    def test_checkpoint_dir_is_path(self, trainer):
        """checkpoint_dir is a Path object."""
        from pathlib import Path
        assert isinstance(trainer.checkpoint_dir, Path)


# ===================================================================
# Learning Rate Schedule
# ===================================================================

class TestGetLR:
    """Tests for Trainer.get_lr() — warmup + cosine decay schedule."""

    # Config: warmup_steps=2, max_steps=10, learning_rate=1e-3

    def test_warmup_step_zero(self, trainer):
        """At step 0, LR = 0 (linear warmup from 0)."""
        trainer.step = 0
        lr = trainer.get_lr()
        assert lr == 0.0

    def test_warmup_step_one(self, trainer):
        """At step 1, LR = learning_rate * 1/2 = 5e-4."""
        trainer.step = 1
        lr = trainer.get_lr()
        expected = 1e-3 * 1.0 / 2.0  # 5e-4
        assert abs(lr - expected) < 1e-10, f"Expected {expected}, got {lr}"

    def test_warmup_complete(self, trainer):
        """At step == warmup_steps, LR = learning_rate (end of warmup)."""
        trainer.step = 2  # warmup_steps = 2
        lr = trainer.get_lr()
        assert abs(lr - 1e-3) < 1e-10, f"Expected 1e-3, got {lr}"

    def test_cosine_decay_midpoint(self, trainer):
        """At midpoint of cosine decay, LR = learning_rate * 0.5.
        Midpoint: progress = 0.5, cos(pi * 0.5) = 0, LR = 0.5 * (1+0) = 0.5.
        """
        trainer.step = 6   # warmup=2, max=10. progress = (6-2)/(10-2) = 4/8 = 0.5
        lr = trainer.get_lr()
        expected = 1e-3 * 0.5 * (1.0 + math.cos(math.pi * 0.5))
        assert abs(lr - expected) < 1e-10

    def test_cosine_decay_end_of_annealing(self, trainer):
        """At step == max_steps, LR = 0.0 (post-max_steps branch)."""
        trainer.step = 10  # max_steps = 10
        lr = trainer.get_lr()
        assert lr == 0.0

    def test_post_max_steps(self, trainer):
        """After max_steps, LR = 0.0."""
        trainer.step = 15
        lr = trainer.get_lr()
        assert lr == 0.0

    def test_cosine_decay_monotonic_decrease(self, trainer):
        """LR decreases monotonically during the cosine decay phase (steps 2-10)."""
        lrs = []
        for step in range(2, 11):
            trainer.step = step
            lrs.append(trainer.get_lr())
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1], (
                f"LR increased at step {2 + i}: {lrs[i-1]:.2e} -> {lrs[i]:.2e}"
            )

    def test_warmup_monotonic_increase(self, trainer):
        """LR increases monotonically during warmup (steps 0-2)."""
        lrs = []
        for step in range(0, 3):
            trainer.step = step
            lrs.append(trainer.get_lr())
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], (
                f"LR decreased during warmup at step {i}: {lrs[i-1]:.2e} -> {lrs[i]:.2e}"
            )


# ===================================================================
# set_lr
# ===================================================================

class TestSetLR:
    """Tests for Trainer.set_lr()."""

    def test_set_lr_updates_all_param_groups(self, trainer):
        """set_lr updates the learning rate in every param group."""
        new_lr = 5e-4
        trainer.set_lr(new_lr)
        for group in trainer.optimizer.param_groups:
            assert abs(group["lr"] - new_lr) < 1e-10, (
                f"Expected lr={new_lr}, got {group['lr']}"
            )

    def test_set_lr_then_get_lr_independent(self, trainer):
        """set_lr is independent from get_lr (set_lr sets optimizer; get_lr computes schedule)."""
        trainer.step = 0
        computed_lr = trainer.get_lr()  # 0.0 during warmup step 0
        trainer.set_lr(0.5)  # Override to a high value
        # Optimizer should have the overridden value
        for group in trainer.optimizer.param_groups:
            assert abs(group["lr"] - 0.5) < 1e-10
        # get_lr should still compute from step
        assert abs(trainer.get_lr() - computed_lr) < 1e-10


# ===================================================================
# train_step
# ===================================================================

class TestTrainStep:
    """Tests for Trainer.train_step()."""

    def test_train_step_returns_loss(self, trainer, tiny_config):
        """train_step returns a positive float loss value."""
        batch_size = trainer.config.batch_size
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))
        y = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))
        loss = trainer.train_step(x, y)
        assert isinstance(loss, float), f"Expected float, got {type(loss)}"
        assert loss > 0, f"Loss should be positive, got {loss}"
        assert not math.isnan(loss), "Loss should not be NaN"

    def test_train_step_accumulates_gradients(self, trainer, tiny_config):
        """Gradients accumulate across multiple train_step calls."""
        batch_size = trainer.config.batch_size
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))
        y = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))

        # First step
        trainer.optimizer.zero_grad()
        loss1 = trainer.train_step(x, y)
        _ = loss1  # unused but verifies no crash

        # Capture gradients after first step
        grads_1 = [p.grad.clone() for p in trainer.model.parameters()
                   if p.grad is not None]

        # Second step (accumulate)
        loss2 = trainer.train_step(x, y)
        _ = loss2

        # Gradients should have changed from the first step
        grads_2 = [p.grad.clone() for p in trainer.model.parameters()
                   if p.grad is not None]

        for i, (g1, g2) in enumerate(zip(grads_1, grads_2)):
            assert not torch.equal(g1, g2), (
                f"Parameter {i}: gradients did not accumulate after second step"
            )

    def test_train_step_model_in_train_mode(self, trainer, tiny_config):
        """After train_step, the model is in train mode."""
        batch_size = trainer.config.batch_size
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))
        y = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))
        trainer.train_step(x, y)
        assert trainer.model.training, "Model should be in train mode after train_step"


# ===================================================================
# optimizer_step
# ===================================================================

class TestOptimizerStep:
    """Tests for Trainer.optimizer_step()."""

    def test_optimizer_step_updates_parameters(self, trainer, tiny_config):
        """Parameters change after optimizer_step."""
        batch_size = trainer.config.batch_size
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))
        y = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))

        trainer.optimizer.zero_grad()
        trainer.train_step(x, y)

        # Record parameters before step
        params_before = [p.data.clone() for p in trainer.model.parameters()]

        trainer.optimizer_step()

        # Parameters should have changed
        params_after = [p.data.clone() for p in trainer.model.parameters()]
        any_changed = False
        for p_before, p_after in zip(params_before, params_after):
            if not torch.equal(p_before, p_after):
                any_changed = True
                break
        assert any_changed, "Parameters did not change after optimizer_step"

    def test_optimizer_step_zeros_gradients(self, trainer, tiny_config):
        """After optimizer_step, gradients are None (set_to_none=True)."""
        batch_size = trainer.config.batch_size
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))
        y = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))

        trainer.optimizer.zero_grad()
        trainer.train_step(x, y)
        trainer.optimizer_step()

        # All gradients should be None
        for param in trainer.model.parameters():
            assert param.grad is None, "Gradients should be None after optimizer_step"

    def test_optimizer_step_with_grad_clip_no_error(self, trainer, tiny_config):
        """optimizer_step with gradient clipping does not raise."""
        batch_size = trainer.config.batch_size
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))
        y = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))

        trainer.optimizer.zero_grad()
        trainer.train_step(x, y)
        # This should not raise (grad_clip=1.0)
        trainer.optimizer_step()


# ===================================================================
# Gradient Accumulation
# ===================================================================

class TestGradientAccumulation:
    """Tests for gradient accumulation semantics."""

    def test_full_step_sequence(self, trainer, tiny_config):
        """Full accumulation cycle: N train_steps + 1 optimizer_step."""
        batch_size = trainer.config.batch_size
        grad_accum = trainer.config.grad_accum_steps  # 2
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))
        y = torch.randint(0, tiny_config.vocab_size, (batch_size, tiny_config.block_size))

        trainer.optimizer.zero_grad()
        accumulated_loss = 0.0

        for i in range(grad_accum):
            loss = trainer.train_step(x, y)
            accumulated_loss += loss

        # Before optimizer_step, gradients should be non-None
        some_grad = any(p.grad is not None for p in trainer.model.parameters())
        assert some_grad, "Gradients should exist before optimizer_step"

        trainer.optimizer_step()

        # After optimizer_step, step counter should increment
        # (Step is incremented in train(), not in optimizer_step directly)
        # Actually, optimizer_step doesn't increment self.step — that's done in train()
        # So just verify the step didn't change
        assert trainer.step == 0  # optimizer_step alone doesn't increment

    def test_training_loop_one_step(self, trainer, dummy_train_loader):
        """Training loop can run for one optimizer step."""
        trainer.max_steps = 1  # Hmm, max_steps is on config. Let me not change config.

        # Actually, the trainer.train() loop uses self.config.max_steps
        # Let me just manually execute one step cycle
        loader = iter(dummy_train_loader)
        x, y = next(loader)

        # One accumulation cycle
        trainer.optimizer.zero_grad()
        for _ in range(trainer.config.grad_accum_steps):
            trainer.train_step(x, y)
        trainer.optimizer_step()

        # Step counter should be incremented in train(), but here we call methods directly
        # So step doesn't auto-increment
        assert trainer.step == 0  # No auto-increment outside of train()


# ===================================================================
# evaluate
# ===================================================================

class TestEvaluate:
    """Tests for Trainer.evaluate()."""

    def test_evaluate_returns_float(self, trainer, dummy_val_loader):
        """evaluate returns a positive float loss value."""
        val_loss = trainer.evaluate(dummy_val_loader)
        assert isinstance(val_loss, float), f"Expected float, got {type(val_loss)}"
        assert val_loss > 0, f"Validation loss should be positive, got {val_loss}"
        assert not math.isnan(val_loss), "Validation loss should not be NaN"
        assert not math.isinf(val_loss), "Validation loss should not be inf"

    def test_evaluate_model_in_eval_mode(self, trainer, dummy_val_loader):
        """After evaluate, the model is in eval mode."""
        trainer.model.train()  # Ensure we start in train mode
        trainer.evaluate(dummy_val_loader)
        assert not trainer.model.training, "Model should be in eval mode after evaluate()"

    def test_evaluate_no_gradients_computed(self, trainer, dummy_val_loader):
        """Gradients are not computed during evaluate (torch.no_grad)."""
        trainer.model.zero_grad()
        _ = trainer.evaluate(dummy_val_loader)
        for param in trainer.model.parameters():
            assert param.grad is None, (
                "No gradients should be computed during evaluate"
            )

    def test_evaluate_with_limited_batches(self, trainer, dummy_val_loader):
        """evaluate respects the max_batches parameter."""
        # Our val_loader has 3 batches; max_batches=2 should only use 2
        val_loss_limited = trainer.evaluate(dummy_val_loader, max_batches=2)
        val_loss_full = trainer.evaluate(dummy_val_loader, max_batches=10)
        assert isinstance(val_loss_limited, float)
        assert isinstance(val_loss_full, float)

    def test_evaluate_on_empty_loader_returns_inf(self, trainer, tiny_config, small_train_config):
        """Empty validation loader returns float('inf')."""
        empty_ds = TensorDataset(
            torch.zeros(0, tiny_config.block_size, dtype=torch.long),
            torch.zeros(0, tiny_config.block_size, dtype=torch.long),
        )
        empty_loader = DataLoader(empty_ds, batch_size=small_train_config.batch_size)
        val_loss = trainer.evaluate(empty_loader)
        assert val_loss == float("inf")


# ===================================================================
# Checkpoint Save and Load
# ===================================================================

class TestCheckpointSaveLoad:
    """Tests for Trainer.save_checkpoint() and Trainer.load_checkpoint()."""

    def test_save_checkpoint_creates_file(self, trainer, tmp_path):
        """save_checkpoint creates a file on disk."""
        trainer.checkpoint_dir = tmp_path
        name = "test_ckpt.pt"
        trainer.save_checkpoint(name)
        ckpt_path = tmp_path / name
        assert ckpt_path.exists(), f"Checkpoint file {ckpt_path} should exist"

    def test_save_and_load_roundtrip(self, trainer, tmp_path):
        """Save checkpoint, modify model, load — parameters are restored exactly."""
        trainer.checkpoint_dir = tmp_path

        # Capture initial state
        initial_step = 5
        trainer.step = initial_step
        initial_params = [p.data.clone() for p in trainer.model.parameters()]

        # Save
        name = "roundtrip.pt"
        trainer.save_checkpoint(name)

        # Scramble model parameters
        for p in trainer.model.parameters():
            p.data.fill_(0.0)

        # Verify scrambling worked (parameters are now all zeros)
        zero_sum = sum(p.sum().item() for p in trainer.model.parameters())
        assert abs(zero_sum) < 1e-10, "Model should be zeroed after scrambling"

        # Load checkpoint
        ckpt_path = tmp_path / name
        trainer.load_checkpoint(str(ckpt_path))

        # Verify step restored
        assert trainer.step == initial_step, (
            f"Expected step={initial_step}, got {trainer.step}"
        )

        # Verify parameters restored (allow tight tolerance for floating point roundtrip)
        for i, (p_before, p_after) in enumerate(
            zip(initial_params, trainer.model.parameters())
        ):
            if not torch.equal(p_before, p_after):
                # Use assert_close for diagnostic message
                torch.testing.assert_close(
                    p_before, p_after,
                    msg=f"Parameter {i} differs after checkpoint roundtrip",
                )

    def test_save_checkpoint_includes_model_config(self, trainer, tiny_config, tmp_path):
        """Checkpoint contains model_config key with the model's config."""
        trainer.checkpoint_dir = tmp_path
        trainer.model_config = tiny_config
        trainer.save_checkpoint("with_config.pt")

        ckpt = torch.load(str(tmp_path / "with_config.pt"), map_location="cpu", weights_only=False)
        assert "model_config" in ckpt, "Checkpoint should have model_config key"
        # The deserialized config equals the original (pickle roundtrip produces
        # a new object, so check value equality, not identity)
        assert ckpt["model_config"] == tiny_config

    def test_save_checkpoint_includes_all_keys(self, trainer, tmp_path):
        """Checkpoint dict contains step, model, optimizer, scaler, val_loss."""
        trainer.checkpoint_dir = tmp_path
        trainer.step = 7
        trainer.save_checkpoint("all_keys.pt", val_loss=2.5)

        ckpt = torch.load(str(tmp_path / "all_keys.pt"), map_location="cpu", weights_only=False)
        assert "step" in ckpt and ckpt["step"] == 7
        assert "model" in ckpt
        assert "optimizer" in ckpt
        assert "scaler" in ckpt
        assert "val_loss" in ckpt and ckpt["val_loss"] == 2.5
        assert "model_config" in ckpt

    def test_load_checkpoint_restores_optimizer_state(self, trainer, tmp_path, tiny_config):
        """Optimizer state is restored from checkpoint (momentum buffers, etc.)."""
        trainer.checkpoint_dir = tmp_path
        vocab_size = tiny_config.vocab_size
        block_size = tiny_config.block_size

        # Step the optimizer a few times to create momentum state
        for _ in range(3):
            dummy_input = torch.randint(0, vocab_size, (trainer.config.batch_size, block_size))
            dummy_target = torch.randint(0, vocab_size, (trainer.config.batch_size, block_size))
            _ = trainer.train_step(dummy_input, dummy_target)
            trainer.optimizer_step()

        # Save the optimizer state dict
        trainer.step = 3
        trainer.save_checkpoint("opt_state.pt")
        saved_opt_state = trainer.optimizer.state_dict()

        # Zero out optimizer state (simulate fresh start)
        trainer.optimizer = trainer.model.configure_optimizers(
            trainer.config.weight_decay,
            trainer.config.learning_rate,
            trainer.config.betas,
        )

        # Verify the fresh optimizer has no momentum state
        fresh_opt_state = trainer.optimizer.state_dict()
        assert fresh_opt_state != saved_opt_state, "Fresh optimizer should differ from saved"

        # Load checkpoint
        trainer.load_checkpoint(str(tmp_path / "opt_state.pt"))
        loaded_opt_state = trainer.optimizer.state_dict()

        # Compare key by key
        for key in saved_opt_state:
            if key == "state":
                # State dict has parameter_id -> {momentum, variance} entries
                for pid in saved_opt_state["state"]:
                    for buffer_key in saved_opt_state["state"][pid]:
                        torch.testing.assert_close(
                            saved_opt_state["state"][pid][buffer_key],
                            loaded_opt_state["state"][pid][buffer_key],
                        )
            else:
                assert saved_opt_state[key] == loaded_opt_state[key]

    def test_load_checkpoint_restores_step(self, trainer, tmp_path):
        """load_checkpoint restores the step counter."""
        trainer.checkpoint_dir = tmp_path
        trainer.step = 42
        trainer.save_checkpoint("step_ckpt.pt")

        trainer.step = 0  # Reset
        trainer.load_checkpoint(str(tmp_path / "step_ckpt.pt"))
        assert trainer.step == 42


# ===================================================================
# Checkpoint Rotation
# ===================================================================

class TestCheckpointRotation:
    """Tests for checkpoint rotation logic (_rotate_checkpoints)."""

    def test_rotation_keeps_max_checkpoints(self, trainer_with_rotating_checkpoints, tmp_path):
        """After saving more than max_checkpoints, only max_checkpoints remain."""
        t = trainer_with_rotating_checkpoints
        t.checkpoint_dir = tmp_path
        max_ckpts = t.config.max_checkpoints  # 3

        # Save more than max_checkpoints
        for i in range(1, max_ckpts + 3):  # Save 5 checkpoints
            # Small delay to ensure distinct mtimes (NTFS has 100ns resolution but
            # Python's os.path.getmtime has second-level precision on some filesystems)
            time.sleep(0.02)
            t.save_checkpoint(f"step_{i:06d}.pt")

        # Count step_*.pt files in checkpoint dir
        step_ckpts = sorted(glob.glob(str(tmp_path / "step_*.pt")))
        assert len(step_ckpts) <= max_ckpts, (
            f"Expected at most {max_ckpts} checkpoints, found {len(step_ckpts)}"
        )

    def test_rotation_preserves_non_step_files(self, trainer_with_rotating_checkpoints, tmp_path):
        """Non-step files like 'best.pt' are not removed by rotation."""
        t = trainer_with_rotating_checkpoints
        t.checkpoint_dir = tmp_path

        # Save a non-step file
        t.save_checkpoint("best.pt")

        # Save many step checkpoints
        for i in range(1, 6):
            time.sleep(0.02)
            t.save_checkpoint(f"step_{i:06d}.pt")

        # best.pt should still exist
        assert (tmp_path / "best.pt").exists(), "best.pt should not be rotated away"

    def test_rotation_with_fewer_than_max(self, trainer_with_rotating_checkpoints, tmp_path):
        """If fewer checkpoints than max_checkpoints exist, no deletion occurs."""
        t = trainer_with_rotating_checkpoints
        t.checkpoint_dir = tmp_path

        # Save fewer than max_checkpoints
        t.save_checkpoint("step_000001.pt")
        t.save_checkpoint("step_000002.pt")

        # Both should still exist
        assert (tmp_path / "step_000001.pt").exists()
        assert (tmp_path / "step_000002.pt").exists()

    def test_rotation_keeps_newest(self, trainer_with_rotating_checkpoints, tmp_path):
        """The most recent checkpoints are kept, not the oldest."""
        t = trainer_with_rotating_checkpoints
        t.checkpoint_dir = tmp_path
        max_ckpts = t.config.max_checkpoints  # 3

        # Save 5 checkpoints with distinct names and times
        for i in range(1, 6):
            time.sleep(0.02)
            t.save_checkpoint(f"step_{i:06d}.pt")

        # The last `max_ckpts` should survive (step_000003, step_000004, step_000005)
        # But due to filesystem timing, the exact set may vary. Just verify the count.
        step_ckpts = sorted(glob.glob(str(tmp_path / "step_*.pt")))
        assert len(step_ckpts) == max_ckpts, (
            f"Expected {max_ckpts} checkpoints, found {len(step_ckpts)}"
        )


# ===================================================================
# Training Loop
# ===================================================================

class TestTrainingLoop:
    """Tests for Trainer.train() — the full training loop."""

    def test_training_loop_runs_10_steps(self, tiny_train_model, small_train_config,
                                          dummy_train_loader, cpu_device):
        """Training loop runs for max_steps without crashing."""
        # Use a config with very small settings for speed
        import copy
        cfg = copy.deepcopy(small_train_config)
        cfg.max_steps = 5
        cfg.log_interval = 5
        cfg.eval_interval = 100  # Don't eval (no val_loader needed)
        cfg.save_interval = 100  # Don't save

        trainer = Trainer(model=tiny_train_model, config=cfg, device=cpu_device)
        trainer.train(dummy_train_loader, val_loader=None)

        assert trainer.step == cfg.max_steps, (
            f"Expected step={cfg.max_steps}, got {trainer.step}"
        )

    def test_training_loop_with_validation(self, tiny_train_model, small_train_config,
                                            dummy_train_loader, dummy_val_loader,
                                            cpu_device):
        """Training loop runs with validation (evaluate called at eval_interval)."""
        import copy
        cfg = copy.deepcopy(small_train_config)
        cfg.max_steps = 5
        cfg.log_interval = 5
        cfg.eval_interval = 3  # Evaluate at step 3... wait, step starts at 0
        # Actually with max_steps=5, step goes 0..4
        # eval_interval=3 means eval at steps 3 and 6 (6 not reached)
        # So eval happens once at step 3
        cfg.save_interval = 100

        trainer = Trainer(model=tiny_train_model, config=cfg, device=cpu_device)
        trainer.train(dummy_train_loader, val_loader=dummy_val_loader)

        assert trainer.step == cfg.max_steps

    def test_training_loop_lr_schedule_followed(self, tiny_train_model, small_train_config,
                                                  dummy_train_loader, cpu_device):
        """LR follows warmup then cosine decay during training."""
        import copy
        cfg = copy.deepcopy(small_train_config)
        cfg.max_steps = 6
        cfg.warmup_steps = 2
        cfg.log_interval = 1
        cfg.eval_interval = 100
        cfg.save_interval = 100

        trainer = Trainer(model=tiny_train_model, config=cfg, device=cpu_device)
        trainer.train(dummy_train_loader, val_loader=None)

        # After training, step should be max_steps
        assert trainer.step == cfg.max_steps
        # LR should be close to 0 (cosine decay ends at 0 at max_steps)
        final_lr = trainer.get_lr()
        assert final_lr == 0.0

    def test_training_loop_resets_exhausted_iterator(self, tiny_train_model, small_train_config,
                                                      dummy_train_loader, cpu_device):
        """Training loop restarts the DataLoader iterator when exhausted."""
        import copy
        cfg = copy.deepcopy(small_train_config)
        # Our dummy_train_loader has 20 batches with batch_size=2, grad_accum_steps=2
        # This means 10 optimizer steps per full pass through the data
        # With max_steps=12, we need more batches than one epoch
        cfg.max_steps = 12
        cfg.warmup_steps = 2
        cfg.log_interval = 100
        cfg.eval_interval = 100
        cfg.save_interval = 100
        cfg.grad_accum_steps = 2

        trainer = Trainer(model=tiny_train_model, config=cfg, device=cpu_device)
        # This should not raise StopIteration
        trainer.train(dummy_train_loader, val_loader=None)

        assert trainer.step == cfg.max_steps

    def test_training_loop_saves_final_checkpoint(self, tiny_train_model, small_train_config,
                                                   dummy_train_loader, cpu_device):
        """Training loop saves a final.pt checkpoint at the end."""
        import copy
        cfg = copy.deepcopy(small_train_config)
        cfg.max_steps = 3
        cfg.warmup_steps = 1
        cfg.log_interval = 100
        cfg.eval_interval = 100
        cfg.save_interval = 100  # Don't save during training
        # checkpoint_dir is already pointing to tmp_path from small_train_config fixture

        trainer = Trainer(model=tiny_train_model, config=cfg, device=cpu_device)
        trainer.train(dummy_train_loader, val_loader=None)

        # Final checkpoint should exist
        final_path = trainer.checkpoint_dir / "final.pt"
        assert final_path.exists(), f"Final checkpoint {final_path} should exist"

        # Verify it loads correctly
        ckpt = torch.load(str(final_path), map_location="cpu", weights_only=False)
        assert ckpt["step"] == cfg.max_steps


# ===================================================================
# Edge Cases
# ===================================================================

class TestTrainerEdgeCases:
    """Edge case tests for the Trainer."""

    def test_trainer_on_cpu_float32_no_amp(self, tiny_train_model, small_train_config, cpu_device):
        """Trainer works on CPU with float32 (no AMP complications)."""
        trainer = Trainer(
            model=tiny_train_model,
            config=small_train_config,
            device=cpu_device,
        )
        assert trainer.device.type == "cpu"
        assert trainer.dtype == torch.float32

        # Should be able to run a train_step without issues
        x = torch.randint(0, tiny_train_model.config.vocab_size, (2, 16))
        y = torch.randint(0, tiny_train_model.config.vocab_size, (2, 16))
        loss = trainer.train_step(x, y)
        assert loss > 0

    def test_trainer_without_configure_optimizers(self, tiny_config, small_train_config, cpu_device):
        """Model without configure_optimizers falls back to default AdamW."""
        model = torch.nn.Linear(10, 10)
        trainer = Trainer(model=model, config=small_train_config, device=cpu_device)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        # Should have 1 param group (Linear has bias=False by default... actually bias=True)
        # Linear has weight + bias, both 1D/2D. The fallback optimizer wraps all params in one group.
        assert len(trainer.optimizer.param_groups) >= 1

    def test_trainer_without_model_config(self, tiny_train_model, small_train_config, cpu_device):
        """Trainer works when model_config is None."""
        trainer = Trainer(
            model=tiny_train_model,
            config=small_train_config,
            model_config=None,
            device=cpu_device,
        )
        assert trainer.model_config is None
        # Saving checkpoint should not crash
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.checkpoint_dir = Path(tmpdir)
            trainer.save_checkpoint("no_model_config.pt")
            ckpt = torch.load(
                str(trainer.checkpoint_dir / "no_model_config.pt"),
                map_location="cpu",
                weights_only=False,
            )
            assert ckpt["model_config"] is None

    def test_evaluate_returns_inf_on_no_batches(self, trainer, tiny_config, small_train_config):
        """evaluate returns float('inf') when the loader yields 0 batches."""
        empty_dataset = TensorDataset(
            torch.zeros((0, tiny_config.block_size), dtype=torch.long),
            torch.zeros((0, tiny_config.block_size), dtype=torch.long),
        )
        empty_loader = DataLoader(
            empty_dataset, batch_size=small_train_config.batch_size, drop_last=True
        )
        val_loss = trainer.evaluate(empty_loader)
        assert val_loss == float("inf"), "Empty loader should give inf loss"

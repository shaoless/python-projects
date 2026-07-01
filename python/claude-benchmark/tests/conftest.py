"""Shared fixtures for all tests.

Fixtures provided:
    - tiny_config: MiniGPTConfig with 2 layers, 32-dim embeddings for fast CPU tests
    - small_train_config: TrainingConfig with float32, small batch, temp dirs
    - dummy_text_file: Path to a small (~3000 char) text file
    - tiny_model: MiniGPT instance using tiny_config (dropout=0, deterministic)
    - tiny_vocab_model: MiniGPT with tiny vocab and layer count for trainer tests
"""

import pytest
import torch
from pathlib import Path

from src.config import MiniGPTConfig, TrainingConfig, LoRAConfig


# ---------------------------------------------------------------------------
# Model configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config() -> MiniGPTConfig:
    """MiniGPT config with tiny dimensions for fast CPU tests.

    - 2 layers, 2 heads, 32-dim embeddings
    - 100 vocab tokens
    - 32 token context
    - Dropout=0 for deterministic behavior
    """
    return MiniGPTConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def tiny_nobias_config() -> MiniGPTConfig:
    """Same as tiny_config but with bias=False."""
    return MiniGPTConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )


@pytest.fixture
def tiny_config_with_dropout() -> MiniGPTConfig:
    """MiniGPT config with dropout > 0 for training-mode tests."""
    return MiniGPTConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.2,
        bias=True,
    )


# ---------------------------------------------------------------------------
# Training configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_train_config(tmp_path) -> TrainingConfig:
    """TrainingConfig for fast CPU tests.

    - float32 dtype (no AMP complications on CPU)
    - Small batch and gradient accumulation
    - All output dirs point to tmp_path
    - num_workers=0 for Windows safety
    """
    return TrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=10,
        warmup_steps=2,
        grad_clip=1.0,
        batch_size=2,
        grad_accum_steps=2,
        dtype="float32",
        compile=False,
        log_interval=5,
        eval_interval=10,
        save_interval=10,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        data_dir=str(tmp_path / "data"),
        num_workers=0,
    )


@pytest.fixture
def medium_train_config(tmp_path) -> TrainingConfig:
    """Larger TrainingConfig for checkpoint rotation tests."""
    return TrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=50,
        warmup_steps=5,
        grad_clip=1.0,
        batch_size=2,
        grad_accum_steps=1,
        dtype="float32",
        compile=False,
        log_interval=100,
        eval_interval=100,
        save_interval=5,
        max_checkpoints=3,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        data_dir=str(tmp_path / "data"),
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# LoRA configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lora_config() -> LoRAConfig:
    """Default LoRAConfig."""
    return LoRAConfig()


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_text_file(tmp_path) -> str:
    """Create a small text file (~3000 chars) for dataset testing."""
    path = tmp_path / "train.txt"
    # ~3000 chars of repeated text
    content = "The quick brown fox jumps over the lazy dog. " * 70
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def dummy_val_file(tmp_path) -> str:
    """Create a small validation text file."""
    path = tmp_path / "val.txt"
    content = "A quick brown fox. " * 50
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def multi_line_text_file(tmp_path) -> str:
    """Create a multi-line text file for prepare_text_for_training testing."""
    path = tmp_path / "multi.txt"
    lines = [
        "This is a very long line that should pass the min_length filter because it has many words.",
        "Short.",
        "Another sufficiently long line for testing the preparation script here.",
        "Tiny.",
        "This line is definitely long enough to be included in the output file.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model(tiny_config):
    """MiniGPT instance with tiny dimensions and dropout=0 for deterministic tests."""
    from src.model import MiniGPT
    model = MiniGPT(tiny_config)
    model.eval()
    return model


@pytest.fixture
def tiny_train_model(tiny_config):
    """MiniGPT in train mode for gradient tests."""
    from src.model import MiniGPT
    model = MiniGPT(tiny_config)
    model.train()
    return model


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def cpu_device() -> torch.device:
    """Always CPU for CI-compatible tests."""
    return torch.device("cpu")

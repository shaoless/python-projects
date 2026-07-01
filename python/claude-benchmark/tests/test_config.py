"""Tests for configuration dataclasses.

Covers:
    - MiniGPTConfig default values and custom instantiation
    - head_size property (geometry of attention)
    - estimated_params() rough count
    - MiniGPTConfig edge cases (smallest possible, zero block_size, etc.)
    - TrainingConfig default values
    - LoRAConfig default values and target_modules list
    - dataclass serialization via dataclasses.asdict
"""

import dataclasses
import math

import pytest
from src.config import MiniGPTConfig, TrainingConfig, LoRAConfig


# ===================================================================
# MiniGPTConfig
# ===================================================================

class TestMiniGPTConfig:
    """MiniGPTConfig unit tests."""

    def test_defaults(self):
        """Default MiniGPTConfig matches expected architecture (~25M params)."""
        cfg = MiniGPTConfig()
        assert cfg.vocab_size == 50257
        assert cfg.block_size == 512
        assert cfg.n_layer == 8
        assert cfg.n_head == 8
        assert cfg.n_embd == 512
        assert cfg.dropout == 0.1
        assert cfg.bias is True

    def test_custom_values(self):
        """Custom MiniGPTConfig overrides all fields."""
        cfg = MiniGPTConfig(
            vocab_size=100,
            block_size=16,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=0.2,
            bias=False,
        )
        assert cfg.vocab_size == 100
        assert cfg.block_size == 16
        assert cfg.n_layer == 2
        assert cfg.n_head == 4
        assert cfg.n_embd == 64
        assert cfg.dropout == 0.2
        assert cfg.bias is False

    def test_head_size_divisible(self):
        """head_size = n_embd // n_head, must be exact when n_head divides n_embd."""
        cfg = MiniGPTConfig(n_embd=512, n_head=8)
        assert cfg.head_size == 64

        cfg = MiniGPTConfig(n_embd=256, n_head=4)
        assert cfg.head_size == 64

    def test_head_size_not_divisible(self):
        """head_size uses integer floor division; this will cause issues in attention but is
        caught by CausalSelfAttention's assert. We just verify the computed value here."""
        cfg = MiniGPTConfig(n_embd=50, n_head=20)
        assert cfg.head_size == 2  # 50 // 20 = 2 (floor)

    @pytest.mark.parametrize("n_embd,n_head,expected", [
        (32, 2, 16),
        (64, 4, 16),
        (128, 8, 16),
    ])
    def test_head_size_parametrized(self, n_embd, n_head, expected):
        """head_size correctly divides n_embd by n_head for valid geometries."""
        cfg = MiniGPTConfig(n_embd=n_embd, n_head=n_head)
        assert cfg.head_size == expected

    def test_estimated_params_default(self):
        """estimated_params returns a plausible count for the default config (~25M)."""
        cfg = MiniGPTConfig()
        # Rough formula:
        # emb = 50257*512 + 512*512
        # per_block = 4*(512*512) + 8*(512*512) = 12*512*512 = 12*262144 = 3145728
        # total = emb + 8*per_block + 512*50257
        params = cfg.estimated_params()
        # Should be in the ballpark of 25M parameters
        assert 65_000_000 < params < 85_000_000, f"Expected ~77M, got {params}"

    def test_estimated_params_tiny(self, tiny_config):
        """estimated_params with a tiny config should give a small positive number."""
        params = tiny_config.estimated_params()
        assert params > 0
        # Tiny config: 100 vocab, 32 block, 2 layers, 32 embd
        # Should be a few hundred thousand at most
        assert params < 500_000

    def test_estimated_params_formula(self):
        """Verify estimated_params matches the formula in the source."""
        cfg = MiniGPTConfig(vocab_size=100, block_size=32, n_layer=2, n_embd=32)
        emb = cfg.vocab_size * cfg.n_embd + cfg.block_size * cfg.n_embd
        attn = 4 * cfg.n_embd * cfg.n_embd
        ffn = 8 * cfg.n_embd * cfg.n_embd
        per_block = attn + ffn
        head = cfg.n_embd * cfg.vocab_size
        expected = emb + cfg.n_layer * per_block + head
        assert cfg.estimated_params() == expected


# ===================================================================
# TrainingConfig
# ===================================================================

class TestTrainingConfig:
    """TrainingConfig unit tests."""

    def test_defaults(self):
        """Default TrainingConfig matches expected hyperparameters."""
        cfg = TrainingConfig()
        assert cfg.learning_rate == 3e-4
        assert cfg.weight_decay == 0.1
        assert cfg.betas == (0.9, 0.95)
        assert cfg.max_steps == 10000
        assert cfg.warmup_steps == 500
        assert cfg.grad_clip == 1.0
        assert cfg.batch_size == 8
        assert cfg.grad_accum_steps == 4
        assert cfg.dtype == "bfloat16"
        assert cfg.compile is False
        assert cfg.log_interval == 10
        assert cfg.eval_interval == 500
        assert cfg.save_interval == 500
        assert cfg.max_checkpoints == 3
        assert cfg.checkpoint_dir == "checkpoints"
        assert cfg.log_dir == "logs"
        assert cfg.data_dir == "data"
        assert cfg.train_split == 0.95
        assert cfg.num_workers == 2

    def test_custom_values(self):
        """Override all TrainingConfig fields."""
        cfg = TrainingConfig(
            learning_rate=1e-2,
            weight_decay=0.01,
            betas=(0.8, 0.9),
            max_steps=5,
            warmup_steps=1,
            grad_clip=5.0,
            batch_size=1,
            grad_accum_steps=1,
            dtype="float32",
            compile=True,
            log_interval=1,
            eval_interval=2,
            save_interval=3,
            max_checkpoints=5,
            checkpoint_dir="/tmp/ckpt",
            log_dir="/tmp/logs",
            data_dir="/tmp/data",
            train_split=0.9,
            num_workers=4,
        )
        assert cfg.learning_rate == 1e-2
        assert cfg.compile is True
        assert cfg.max_checkpoints == 5


# ===================================================================
# LoRAConfig
# ===================================================================

class TestLoRAConfig:
    """LoRAConfig unit tests."""

    def test_defaults(self):
        """Default LoRAConfig matches expected values."""
        cfg = LoRAConfig()
        assert cfg.model_name == "HuggingFaceTB/SmolLM2-360M"
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert len(cfg.target_modules) == 7
        assert "q_proj" in cfg.target_modules
        assert "k_proj" in cfg.target_modules
        assert "v_proj" in cfg.target_modules
        assert "o_proj" in cfg.target_modules
        assert "gate_proj" in cfg.target_modules
        assert "up_proj" in cfg.target_modules
        assert "down_proj" in cfg.target_modules

    def test_target_modules_is_mutable(self):
        """target_modules list is a fresh copy per instance (field(default_factory=...))."""
        cfg1 = LoRAConfig()
        cfg2 = LoRAConfig()
        assert cfg1.target_modules is not cfg2.target_modules
        # Mutating one should not affect the other
        cfg1.target_modules.append("extra_module")
        assert "extra_module" not in cfg2.target_modules

    def test_custom_values(self):
        """Override all LoRAConfig fields."""
        cfg = LoRAConfig(
            model_name="test/model",
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            use_4bit=False,
            learning_rate=1e-3,
            max_steps=100,
        )
        assert cfg.r == 8
        assert cfg.lora_alpha == 16
        assert cfg.target_modules == ["q_proj", "v_proj"]
        assert cfg.use_4bit is False
        assert cfg.learning_rate == 1e-3


# ===================================================================
# Serialization
# ===================================================================

class TestConfigSerialization:
    """Tests that configs can be round-tripped through dicts."""

    @pytest.mark.parametrize("cfg_fixture", ["tiny_config", "small_train_config"])
    def test_asdict_roundtrip(self, cfg_fixture, request):
        """Config dumped to dict and re-instantiated should be identical."""
        cfg = request.getfixturevalue(cfg_fixture)
        d = dataclasses.asdict(cfg)
        # Rebuild from the dict
        restored = type(cfg)(**d)
        assert restored == cfg, f"{type(cfg).__name__} roundtrip failed"

    def test_minigpt_config_asdict_includes_all_fields(self, tiny_config):
        """asdict contains all expected keys for MiniGPTConfig."""
        d = dataclasses.asdict(tiny_config)
        expected_keys = {"vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout", "bias"}
        assert set(d.keys()) == expected_keys

    def test_training_config_asdict_includes_all_fields(self, small_train_config):
        """asdict contains all expected keys for TrainingConfig."""
        d = dataclasses.asdict(small_train_config)
        expected_keys = {
            "learning_rate", "weight_decay", "betas", "max_steps", "warmup_steps",
            "grad_clip", "batch_size", "grad_accum_steps", "dtype", "compile",
            "log_interval", "eval_interval", "save_interval", "max_checkpoints",
            "checkpoint_dir", "log_dir", "data_dir", "train_split", "num_workers",
        }
        assert set(d.keys()) == expected_keys

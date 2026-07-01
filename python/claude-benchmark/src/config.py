"""Model and training configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MiniGPTConfig:
    """Configuration for the Mini-GPT model (~25M parameters)."""

    # Vocabulary (must match tokenizer: GPT-2 = 50257)
    vocab_size: int = 50257

    # Context length
    block_size: int = 512

    # Transformer architecture (~25M parameters total)
    n_layer: int = 8        # Number of transformer blocks
    n_head: int = 8         # Number of attention heads
    n_embd: int = 512       # Embedding dimension

    # Regularization
    dropout: float = 0.1
    bias: bool = True       # Whether to use bias in linear layers

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    def estimated_params(self) -> int:
        """Rough estimate of total parameters."""
        # Embedding: vocab * embd + block_size * embd
        emb = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        # Per block: 4 * (embd * embd) for attention + 8 * (embd * embd) for FFN
        attn = 4 * self.n_embd * self.n_embd
        ffn = 8 * self.n_embd * self.n_embd  # MLP has expansion factor 4, so 2 * 4 * embd^2
        per_block = attn + ffn
        # Final layer norm + lm_head
        head = self.n_embd * self.vocab_size
        return emb + self.n_layer * per_block + head


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_steps: int = 10000
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Batch
    batch_size: int = 8
    grad_accum_steps: int = 4  # Effective batch = 8 * 4 = 32

    # Precision
    dtype: str = "bfloat16"    # "float32", "bfloat16", "float16"
    compile: bool = False      # torch.compile (PyTorch 2.0+)

    # Logging & checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 500
    max_checkpoints: int = 3   # Keep only the N most recent

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_dir: str = "data"

    # Dataset
    train_split: float = 0.95
    num_workers: int = 2


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    # Base model
    model_name: str = "HuggingFaceTB/SmolLM2-360M"
    # Alternative: "Qwen/Qwen2.5-0.5B"

    # LoRA hyperparameters
    r: int = 16                # Rank
    lora_alpha: int = 32       # Scaling factor
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Quantization
    use_4bit: bool = True      # QLoRA with 4-bit quantization
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 2000
    warmup_steps: int = 100

    # Sequence
    max_length: int = 512

    # Paths
    output_dir: str = "checkpoints/lora-adapter"
    log_dir: str = "logs/lora"

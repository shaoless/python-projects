"""Mini-GPT: A GPT-2 style Transformer Decoder in pure PyTorch.

This is a minimal but complete implementation designed for learning.
Every component is hand-written — no transformers library used.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Causal means tokens can only attend to previous tokens (no peeking ahead).
    """

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Q, K, V projections in one big matrix for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Causal mask (registered as buffer so it moves with the model)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        y = att @ v  # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        y = F.dropout(y, p=self.dropout, training=self.training)
        return y


# ---------------------------------------------------------------------------
# MLP / FeedForward
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Feed-forward network with GELU activation.

    Architecture: Linear → GELU → Linear (expansion factor 4).
    Uses the "approx='tanh'" variant of GELU matching GPT-2.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """One transformer block: pre-norm attention + pre-norm MLP.

    Uses pre-layer-norm (norm before sublayer, not after) — the modern standard.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# MiniGPT
# ---------------------------------------------------------------------------

class MiniGPT(nn.Module):
    """Mini-GPT: a GPT-2 style language model.

    Architecture:
        1. Token + Position embeddings
        2. N layers of TransformerBlock (decoder-only)
        3. Final LayerNorm
        4. Language modeling head (Linear → vocab)

    ~25M parameters with the default config.

    Usage:
        config = MiniGPTConfig()
        model = MiniGPT(config)
        logits = model(input_ids)  # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)      # Token
        self.wpe = nn.Embedding(config.block_size, config.n_embd)      # Position

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output head (tied with token embeddings for efficiency)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: share weights between embedding and output
        self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"MiniGPT: {n_params/1e6:.1f}M parameters")

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay only on 2D+ parameters."""
        # Separate parameters: apply weight decay to matmuls (ndim >= 2),
        # no weight decay to biases and layer norms (ndim < 2)
        decay_params = []
        nodecay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        print(f"Optimizer: {len(decay_params)} decay params, {len(nodecay_params)} no-decay params")

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 convention."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: (B, T) tensor of token IDs.
            targets: (B, T) tensor of target token IDs (shifted input for next-token prediction).

        Returns:
            (logits, loss):
                logits: (B, T, vocab_size) raw scores
                loss: cross-entropy loss if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.block_size, f"Input length {T} exceeds block_size {self.config.block_size}"

        # Token + Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)  # (1, T)
        tok_emb = self.wte(input_ids)    # (B, T, C)
        pos_emb = self.wpe(pos)          # (1, T, C)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Auto-regressive text generation.

        Args:
            input_ids: (1, T) starting token sequence.
            max_new_tokens: How many tokens to generate.
            temperature: Softmax temperature (>1 = more random, <1 = more deterministic).
            top_k: Only sample from top-k candidates.
            top_p: Nucleus sampling — sample from the smallest set with cumulative prob >= top_p.

        Returns:
            (1, T + max_new_tokens) — the full sequence including the prompt.
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to block_size for long sequences
            x_crop = input_ids[:, -self.config.block_size:]

            logits, _ = self(x_crop)
            # Take logits for the last position
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift so we keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids

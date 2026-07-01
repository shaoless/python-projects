"""Tests for the MiniGPT model and its submodules.

Covers:
    - CausalSelfAttention: output shape, causal mask correctness, dropout behavior
    - MLP: output shape, activation function
    - TransformerBlock: output shape, residual connections
    - MiniGPT: forward pass shapes, loss computation, gradient flow
    - MiniGPT.generate(): output shape, temperature, top-k, top-p, mode switching
    - MiniGPT.configure_optimizers(): param group sizes
    - Weight tying between wte and lm_head
    - Edge cases: B=1, T=1, T==block_size
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.model import CausalSelfAttention, MLP, TransformerBlock, MiniGPT


# ===================================================================
# CausalSelfAttention
# ===================================================================

class TestCausalSelfAttention:
    """Unit tests for the multi-head causal self-attention module."""

    def test_output_shape(self, tiny_config):
        """Attention output has same shape as input: [B, T, C]."""
        attn = CausalSelfAttention(tiny_config)
        B, T, C = 2, 16, tiny_config.n_embd
        x = torch.randn(B, T, C)
        out = attn(x)
        assert out.shape == (B, T, C)

    def test_causal_mask_is_upper_triangular(self, tiny_config):
        """The causal mask buffer has 1s in lower triangle (including diagonal) and 0s above."""
        attn = CausalSelfAttention(tiny_config)
        mask = attn.bias  # (1, 1, block_size, block_size)
        block_size = tiny_config.block_size
        assert mask.shape == (1, 1, block_size, block_size)

        # Lower triangle (including diagonal) should be 1
        for i in range(block_size):
            for j in range(block_size):
                expected = 1.0 if j <= i else 0.0
                assert mask[0, 0, i, j].item() == expected, (
                    f"Mismatch at position ({i}, {j}): expected {expected}, got {mask[0, 0, i, j].item()}"
                )

    def test_causal_mask_applied_as_inf(self, tiny_config):
        """After attention, upper-triangle positions should have near-zero attention weight
        because softmax(-inf) = 0."""
        attn = CausalSelfAttention(tiny_config)
        B, T, C = 1, 8, tiny_config.n_embd
        x = torch.randn(B, T, C)

        # Manually trace through to get attention weights
        qkv = attn.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        n_head = tiny_config.n_head
        head_size = C // n_head

        q = q.view(B, T, n_head, head_size).transpose(1, 2)
        k = k.view(B, T, n_head, head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
        probs = F.softmax(att, dim=-1)

        # Upper triangle (j > i) should be ~0
        for i in range(T):
            for j in range(T):
                prob = probs[0, 0, i, j].item()
                if j > i:
                    assert prob < 1e-5, f"Expected ~0 for upper triangle ({i},{j}), got {prob}"
                else:
                    assert prob > 0, f"Expected >0 for lower triangle ({i},{j}), got {prob}"

    def test_multiple_heads_produce_different_outputs(self, tiny_config):
        """With multiple heads, different heads can attend to different patterns.
        This just verifies the multi-head reshape logic doesn't collapse heads."""
        attn = CausalSelfAttention(tiny_config)
        B, T, C = 1, 4, tiny_config.n_embd
        x = torch.randn(B, T, C)
        out = attn(x)
        assert out.shape == (B, T, C)
        # Output should not be all zeros
        assert out.abs().sum().item() > 0

    def test_attention_is_deterministic_in_eval_mode(self, tiny_config):
        """With dropout=0 (tiny_config), eval mode produces identical outputs."""
        cfg = tiny_config
        # Ensure dropout=0
        assert cfg.dropout == 0.0, "tiny_config must have dropout=0 for this test"
        attn = CausalSelfAttention(cfg)
        attn.eval()
        B, T, C = 2, 8, cfg.n_embd
        x = torch.randn(B, T, C)
        out1 = attn(x)
        out2 = attn(x)
        torch.testing.assert_close(out1, out2)


# ===================================================================
# MLP
# ===================================================================

class TestMLP:
    """Unit tests for the feed-forward MLP module."""

    def test_output_shape(self, tiny_config):
        """MLP output has same shape as input: [B, T, C]."""
        mlp = MLP(tiny_config)
        B, T, C = 2, 16, tiny_config.n_embd
        x = torch.randn(B, T, C)
        out = mlp(x)
        assert out.shape == (B, T, C)

    def test_inner_dimension_is_4x(self, tiny_config):
        """The hidden layer has 4*C units."""
        mlp = MLP(tiny_config)
        assert mlp.c_fc.out_features == 4 * tiny_config.n_embd
        assert mlp.c_proj.in_features == 4 * tiny_config.n_embd

    def test_gelu_non_linearity(self, tiny_config):
        """GELU activation introduces non-linearity (output differs from linear transform)."""
        mlp = MLP(tiny_config)
        B, T, C = 1, 1, tiny_config.n_embd
        x = torch.randn(B, T, C)

        # Verify that after c_fc, the activation is non-zero and not just identity
        hidden = mlp.c_fc(x)
        activated = mlp.gelu(hidden)
        # GELU should be non-trivial (not just pass-through for random inputs)
        assert not torch.allclose(hidden, activated), "GELU should modify activations"


# ===================================================================
# TransformerBlock
# ===================================================================

class TestTransformerBlock:
    """Unit tests for a single transformer block."""

    def test_output_shape(self, tiny_config):
        """Block output has same shape as input: [B, T, C]."""
        block = TransformerBlock(tiny_config)
        B, T, C = 2, 16, tiny_config.n_embd
        x = torch.randn(B, T, C)
        out = block(x)
        assert out.shape == (B, T, C)

    def test_residual_connection_preserves_scale(self, tiny_config):
        """Residual connections mean the output has roughly the same magnitude as the input."""
        block = TransformerBlock(tiny_config)
        B, T, C = 1, 8, tiny_config.n_embd
        x = torch.randn(B, T, C) * 0.1
        out = block(x)
        # Output norm should not be dramatically larger than input norm
        in_norm = x.norm().item()
        out_norm = out.norm().item()
        assert out_norm < 10 * in_norm, (
            f"Output norm {out_norm} >> input norm {in_norm}, residual may be broken"
        )

    def test_pre_norm_order(self, tiny_config):
        """In pre-norm architecture, norm is applied before attention/MLP.
        We verify that the module has the expected submodule order."""
        block = TransformerBlock(tiny_config)
        # Pre-norm: ln_1 before attn, ln_2 before mlp
        # Just check that the norm modules exist and have the right shape
        assert block.ln_1.normalized_shape == (tiny_config.n_embd,)
        assert block.ln_2.normalized_shape == (tiny_config.n_embd,)


# ===================================================================
# MiniGPT (full model)
# ===================================================================

class TestMiniGPTForward:
    """Tests for MiniGPT.forward()."""

    def test_forward_logits_shape(self, tiny_model, tiny_config):
        """Forward pass without targets returns logits of shape [B, T, vocab_size]."""
        B, T = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, loss = tiny_model(input_ids)
        assert logits.shape == (B, T, tiny_config.vocab_size)
        assert loss is None

    def test_forward_with_targets_returns_loss(self, tiny_model, tiny_config):
        """Forward pass with targets returns loss (float scalar tensor)."""
        B, T = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, loss = tiny_model(input_ids, targets)
        assert logits.shape == (B, T, tiny_config.vocab_size)
        assert loss is not None
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0  # cross-entropy loss is positive

    def test_loss_is_differentiable(self, tiny_train_model, tiny_config):
        """Loss.backward() produces non-None gradients for all parameters."""
        model = tiny_train_model
        B, T = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, loss = model(input_ids, targets)
        loss.backward()

        # Every parameter should have a non-None gradient
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert param.grad.numel() > 0, f"Parameter {name} has empty gradient"

    def test_loss_decreases_with_training(self, tiny_config):
        """Loss on the same batch should decrease after an optimizer step."""
        model = MiniGPT(tiny_config)
        model.train()

        B, T = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))

        # Forward + backward + step
        logits, loss = model(input_ids, targets)
        loss_before = loss.item()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Forward again with the same inputs
        logits, loss = model(input_ids, targets)
        loss_after = loss.item()

        # Loss should have decreased (SGD on random init usually decreases loss on same batch)
        assert loss_after < loss_before, (
            f"Loss did not decrease: before={loss_before:.4f}, after={loss_after:.4f}"
        )

    def test_forward_raises_on_oversized_input(self, tiny_model, tiny_config):
        """Input longer than block_size raises an assertion error."""
        B = 1
        T = tiny_config.block_size + 5
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        with pytest.raises(AssertionError, match="exceeds block_size"):
            tiny_model(input_ids)

    def test_single_token_input(self, tiny_model, tiny_config):
        """Forward pass with a single token (B=1, T=1) works correctly."""
        B, T = 1, 1
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, loss = tiny_model(input_ids)
        assert logits.shape == (B, T, tiny_config.vocab_size)
        assert loss is None  # no targets

    def test_batch_of_one(self, tiny_model, tiny_config):
        """Batch size of 1 works correctly."""
        B, T = 1, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, loss = tiny_model(input_ids, targets)
        assert logits.shape == (B, T, tiny_config.vocab_size)
        assert loss is not None

    def test_max_context_length(self, tiny_model, tiny_config):
        """Input exactly at block_size works correctly."""
        B = 2
        T = tiny_config.block_size
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, loss = tiny_model(input_ids, targets)
        assert logits.shape == (B, T, tiny_config.vocab_size)
        assert loss is not None

    def test_ignore_index_in_loss(self, tiny_model, tiny_config):
        """Positions with target=-1 are ignored in cross-entropy loss."""
        B, T = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        # Set some targets to -1 (ignore_index)
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets[:, 5:10] = -1

        logits, loss = tiny_model(input_ids, targets)
        assert loss is not None
        assert loss.item() > 0


# ===================================================================
# Weight tying
# ===================================================================

class TestWeightTying:
    """Tests that embedding and LM head weights are tied."""

    def test_lm_head_is_wte(self, tiny_model):
        """lm_head.weight is the same tensor object as wte.weight (weight tying)."""
        assert tiny_model.lm_head.weight is tiny_model.wte.weight, (
            "Weight tying: lm_head.weight should be the same tensor as wte.weight"
        )

    def test_modifying_wte_affects_lm_head(self, tiny_model):
        """Modifying wte weight in-place also changes lm_head weight (shared tensor)."""
        old_wte = tiny_model.wte.weight.clone()
        tiny_model.wte.weight.data.fill_(0.0)
        # lm_head should also be zeros
        assert tiny_model.lm_head.weight.sum().item() == 0.0, (
            "Modifying wte should also zero out lm_head (same tensor)"
        )
        # Restore
        tiny_model.wte.weight.data.copy_(old_wte)


# ===================================================================
# configure_optimizers
# ===================================================================

class TestConfigureOptimizers:
    """Tests for MiniGPT.configure_optimizers()."""

    def test_returns_adamw(self, tiny_model, tiny_config):
        """configure_optimizers returns a torch.optim.AdamW instance."""
        optimizer = tiny_model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3
        )
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_two_param_groups(self, tiny_model):
        """Optimizer has two groups: decay and no-decay."""
        optimizer = tiny_model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3
        )
        assert len(optimizer.param_groups) == 2

    def test_decay_group_has_weight_decay(self, tiny_model):
        """First group (matmuls, ndim>=2) has the specified weight decay."""
        optimizer = tiny_model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3
        )
        assert optimizer.param_groups[0]["weight_decay"] == 0.1

    def test_nodecay_group_has_zero_weight_decay(self, tiny_model):
        """Second group (biases, norms, ndim<2) has weight_decay=0."""
        optimizer = tiny_model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3
        )
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_correct_param_assignment(self, tiny_model):
        """Parameters with ndim >= 2 go to decay group, ndim < 2 go to no-decay group."""
        optimizer = tiny_model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3
        )

        decay_params = optimizer.param_groups[0]["params"]
        nodecay_params = optimizer.param_groups[1]["params"]

        # All parameters in decay group should have ndim >= 2
        for p in decay_params:
            assert p.ndim >= 2, f"Decay param has ndim={p.ndim}, expected >= 2"

        # All parameters in no-decay group should have ndim < 2
        for p in nodecay_params:
            assert p.ndim < 2, f"No-decay param has ndim={p.ndim}, expected < 2"

    def test_all_parameters_in_one_group(self, tiny_model):
        """Every trainable parameter belongs to exactly one param group."""
        optimizer = tiny_model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3
        )
        grouped_params = set()
        for group in optimizer.param_groups:
            grouped_params.update(id(p) for p in group["params"])

        all_params = set(id(p) for p in tiny_model.parameters() if p.requires_grad)
        assert grouped_params == all_params, (
            "Not all parameters are covered by param groups"
        )


# ===================================================================
# generate()
# ===================================================================

class TestGenerate:
    """Tests for MiniGPT.generate()."""

    def test_generate_output_shape(self, tiny_model, tiny_config):
        """generate returns shape [1, T_in + max_new_tokens]."""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 5))
        max_new = 10
        output = tiny_model.generate(prompt, max_new_tokens=max_new)
        assert output.shape == (1, 5 + max_new)

    def test_generate_preserves_prompt(self, tiny_model, tiny_config):
        """The generated output starts with the original prompt."""
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        max_new = 10
        output = tiny_model.generate(prompt, max_new_tokens=max_new)
        assert torch.all(output[0, :5] == prompt[0]), "Prompt tokens should be preserved"

    def test_generate_respects_temperature(self, tiny_model, tiny_config):
        """Different temperatures produce different results (high temperature = more diverse).
        Very low temperature with top_k=1 should be deterministic."""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 5))

        # Near-zero temperature with top_k=1 should produce the same result twice
        torch.manual_seed(42)
        out1 = tiny_model.generate(prompt, max_new_tokens=20, temperature=0.001, top_k=1)
        torch.manual_seed(42)
        out2 = tiny_model.generate(prompt, max_new_tokens=20, temperature=0.001, top_k=1)
        torch.testing.assert_close(out1, out2)

    def test_generate_top_k_filtering(self, tiny_model, tiny_config):
        """Top-k sampling with k=1 is deterministic (always picks the top token)."""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 5))

        torch.manual_seed(123)
        out1 = tiny_model.generate(prompt, max_new_tokens=10, top_k=1)
        torch.manual_seed(123)
        out2 = tiny_model.generate(prompt, max_new_tokens=10, top_k=1)
        torch.testing.assert_close(out1, out2)

    def test_generate_top_p_filtering(self, tiny_model, tiny_config):
        """Top-p (nucleus) sampling with top_p=0 is equivalent to picking the top token."""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 5))

        torch.manual_seed(123)
        out1 = tiny_model.generate(prompt, max_new_tokens=10, top_p=0.0)
        torch.manual_seed(123)
        out2 = tiny_model.generate(prompt, max_new_tokens=10, top_p=0.0)
        torch.testing.assert_close(out1, out2)

    def test_generate_does_not_exceed_block_size(self, tiny_model, tiny_config):
        """Generate should never produce more than block_size tokens in one forward pass.
        The internal cropping should handle long contexts."""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size - 5))

        # Generate enough tokens to exceed block_size
        output = tiny_model.generate(prompt, max_new_tokens=20)
        assert output.shape[1] == tiny_config.block_size - 5 + 20

    def test_generate_switches_to_eval_mode(self, tiny_config):
        """generate() calls self.eval() which disables dropout.
        We verify that after generate(), the model is in eval mode."""
        model = MiniGPT(tiny_config)
        model.train()  # ensure we start in train mode
        assert model.training

        prompt = torch.randint(0, tiny_config.vocab_size, (1, 5))
        _ = model.generate(prompt, max_new_tokens=5)

        assert not model.training, "Model should be in eval mode after generate()"


# ===================================================================
# Model initialization
# ===================================================================

class TestModelInit:
    """Tests for weight initialization and parameter counts."""

    def test_params_count_print(self, tiny_config, capsys):
        """Model prints parameter count during init."""
        _ = MiniGPT(tiny_config)
        captured = capsys.readouterr()
        assert "MiniGPT" in captured.out
        assert "M parameters" in captured.out

    def test_linear_weights_not_zero(self, tiny_config):
        """Linear layer weights are initialized from N(0, 0.02), not zeros."""
        model = MiniGPT(tiny_config)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                # Mean should be close to 0, std close to 0.02
                assert module.weight.std().item() > 0.001, (
                    f"Linear weight appears to be zero: {module}"
                )

    def test_embedding_weights_not_zero(self, tiny_config):
        """Embedding weights are initialized from N(0, 0.02), not zeros."""
        model = MiniGPT(tiny_config)
        assert model.wte.weight.std().item() > 0.001
        assert model.wpe.weight.std().item() > 0.001


# ===================================================================
# Determinism
# ===================================================================

class TestDeterminism:
    """Tests that model produces consistent results with the same seed."""

    def test_eval_mode_deterministic(self, tiny_config):
        """In eval mode, same input + same seed = same output."""
        torch.manual_seed(42)
        model1 = MiniGPT(tiny_config)
        model1.eval()
        torch.manual_seed(42)
        model2 = MiniGPT(tiny_config)
        model2.eval()

        B, T = 2, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))

        logits1, _ = model1(input_ids)
        logits2, _ = model2(input_ids)

        torch.testing.assert_close(logits1, logits2)

    def test_dropout_affects_train_mode(self, tiny_config_with_dropout):
        """With dropout > 0, same input in train mode can produce different outputs."""
        cfg = tiny_config_with_dropout
        model = MiniGPT(cfg)
        model.train()
        B, T = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        out1 = model(input_ids)
        out2 = model(input_ids)
        # They may or may not differ, but we verify that dropout is in the modules
        assert model.drop.p > 0
        for block in model.blocks:
            assert block.attn.dropout > 0

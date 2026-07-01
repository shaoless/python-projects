#!/usr/bin/env python
"""Generate text from a trained Mini-GPT or LoRA-adapter model.

Usage:
    # Generate from Mini-GPT checkpoint:
    python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"

    # Generate from LoRA adapter:
    python scripts/generate.py --lora-adapter checkpoints/lora-adapter/adapter --prompt "Explain AI:"

    # Control generation:
    python scripts/generate.py --checkpoint checkpoints/final.pt --prompt "The cat" --temperature 0.8 --max_tokens 200
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_from_checkpoint(
    checkpoint_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> None:
    """Generate text from a Mini-GPT checkpoint."""
    from src.config import MiniGPTConfig
    from src.model import MiniGPT
    import tiktoken

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_config", MiniGPTConfig())

    # Create model
    model = MiniGPT(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Encode prompt
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

    step = ckpt.get('step', 'N/A')
    val_loss = ckpt.get('val_loss')
    loss_str = f"{val_loss:.4f}" if val_loss is not None else 'N/A'
    print(f"Model step: {step} | val_loss: {loss_str}")
    print("-" * 40)
    print(f"Prompt: {prompt}")
    print("-" * 40)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    generated = enc.decode(output[0].tolist())
    print(generated)
    print("-" * 40)


def generate_from_lora(
    adapter_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> None:
    """Generate text from a LoRA fine-tuned model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading LoRA adapter from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"Prompt: {prompt}")
    print("-" * 40)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated)
    print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Mini-GPT checkpoint (.pt)")
    parser.add_argument("--lora-adapter", type=str, default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Input prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling filter")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling filter")
    args = parser.parse_args()

    if args.checkpoint:
        generate_from_checkpoint(
            args.checkpoint,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )
    elif args.lora_adapter:
        generate_from_lora(
            args.lora_adapter,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )
    else:
        print("ERROR: Provide --checkpoint or --lora-adapter")
        sys.exit(1)


if __name__ == "__main__":
    main()

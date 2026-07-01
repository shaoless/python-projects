"""Train and use a BPE tokenizer via tiktoken."""

import json
import os
from pathlib import Path
from typing import Optional

import tiktoken
from tiktoken.load import load_tiktoken_bpe


def train_tokenizer(
    text_file: str,
    vocab_size: int = 16384,
    special_tokens: Optional[list[str]] = None,
) -> bytes:
    """Train a BPE tokenizer on a text file using tiktoken.

    Args:
        text_file: Path to a plain text file (one document per line recommended).
        vocab_size: Target vocabulary size.
        special_tokens: Special tokens to register (e.g. <|endoftext|>, <|pad|>).

    Returns:
        The BPE merge ranks as bytes (can be saved to .tiktoken file).
    """
    if special_tokens is None:
        special_tokens = [
            "<|endoftext|>",
            "<|pad|>",
            "<|unk|>",
        ]

    # Read training data — tiktoken expects UTF-8 text
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Train BPE
    encoding = tiktoken.Encoding(
        name="minigpt_bpe",
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks={},  # Start empty, will be filled
        special_tokens={tok: vocab_size + i for i, tok in enumerate(special_tokens)},
    )

    # Actually train: use the GPT-2 base encoding's BPE
    # tiktoken doesn't expose a direct train API — we fall back to using GPT-2 tokenizer
    # and extend it, OR we use HuggingFace tokenizers for custom BPE training.
    #
    # For simplicity, we use the GPT-2 base BPE (cl100k_base or o200k_base) which covers
    # most use-cases. For a truly custom tokenizer, see the train_hf_tokenizer() below.
    raise NotImplementedError(
        "tiktoken does not support training from scratch. "
        "Use train_hf_tokenizer() for custom BPE training, "
        "or use get_pretrained_tokenizer() for GPT-2/4 tokenizers."
    )


def train_hf_tokenizer(
    text_file: str,
    save_path: str,
    vocab_size: int = 16384,
) -> None:
    """Train a BPE tokenizer using HuggingFace tokenizers library.

    This is the recommended path for custom tokenizer training.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train
    tokenizer.train(files=[text_file], trainer=trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})")


def get_pretrained_tokenizer(name: str = "gpt2") -> tiktoken.Encoding:
    """Get a pretrained tokenizer.

    Common options:
        - "gpt2"     → GPT-2 tokenizer (vocab 50257)
        - "cl100k_base" → GPT-4 tokenizer (vocab 100277)
        - "o200k_base"  → GPT-4o tokenizer (vocab 200k)

    For training Mini-GPT from scratch, use "gpt2" as it's the smallest.
    """
    return tiktoken.get_encoding(name)


def load_hf_tokenizer(path: str):
    """Load a HuggingFace tokenizer from disk."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def prepare_text_for_training(
    input_files: list[str],
    output_file: str,
    min_length: int = 50,
) -> None:
    """Concatenate and clean text files into one training file.

    Each non-empty line becomes a training sample.
    """
    total_lines = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for file_path in input_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) >= min_length:
                        out.write(line + "\n")
                        total_lines += 1

    print(f"Prepared {total_lines} lines → {output_file}")

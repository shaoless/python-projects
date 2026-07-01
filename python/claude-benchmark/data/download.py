"""Download and prepare datasets for training.

Usage:
    python data/download.py wikitext       # Download WikiText-2
    python data/download.py tinystories    # Download TinyStories
    python data/download.py alpaca         # Download Alpaca instruction data
    python data/download.py all            # Download everything
"""

import argparse
import os
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent

def download_wikitext() -> None:
    """Download WikiText-2 dataset for language model training."""
    print("Downloading WikiText-2...")
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"wikitext_{split}.txt"
        texts = [item["text"] for item in dataset[split] if item["text"].strip()]
        with open(path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.replace("\n", " ").strip() + "\n")
        print(f"  {split}: {len(texts)} docs → {path}")

    print("WikiText-2 ready.")


def download_tinystories() -> None:
    """Download TinyStories dataset (simple stories for small models)."""
    print("Downloading TinyStories...")
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories")

    for split in ["train", "validation"]:
        path = DATA_DIR / f"tinystories_{split}.txt"
        texts = [item["text"] for item in dataset[split] if item["text"].strip()]
        with open(path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.replace("\n", " ").strip() + "\n")
        print(f"  {split}: {len(texts)} docs → {path}")

    print("TinyStories ready.")


def download_alpaca() -> None:
    """Download Alpaca instruction fine-tuning dataset."""
    print("Downloading Alpaca (cleaned)...")
    from datasets import load_dataset

    dataset = load_dataset("yahma/alpaca-cleaned")

    for split in ["train"]:
        path = DATA_DIR / f"alpaca_{split}.json"
        data = dataset[split]
        data.to_json(str(path))
        print(f"  {split}: {len(data)} examples → {path}")

    # Also save as plain text for language modeling
    path = DATA_DIR / "alpaca_train.txt"
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset["train"]:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            if instruction and output:
                f.write(f"Instruction: {instruction}\nResponse: {output}\n\n")

    print("Alpaca ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument(
        "dataset",
        choices=["wikitext", "tinystories", "alpaca", "all"],
        help="Which dataset to download",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.dataset in ("wikitext", "all"):
        download_wikitext()

    if args.dataset in ("tinystories", "all"):
        download_tinystories()

    if args.dataset in ("alpaca", "all"):
        download_alpaca()

    print("\nAll done!")


if __name__ == "__main__":
    main()

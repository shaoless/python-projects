"""Text dataset utilities for language model training."""

import torch
from torch.utils.data import Dataset, IterableDataset
import tiktoken


class TextDataset(Dataset):
    """Map-style dataset: loads tokenized text and returns (x, y) pairs.

    Each sample is a contiguous block of block_size tokens.
    x = tokens[0:block_size], y = tokens[1:block_size+1] (shifted by 1 for next-token prediction).
    """

    def __init__(
        self,
        file_path: str,
        block_size: int,
        encoding_name: str = "gpt2",
        max_chars: int = 0,
    ) -> None:
        self.block_size = block_size
        self.enc = tiktoken.get_encoding(encoding_name)

        with open(file_path, "r", encoding="utf-8") as f:
            if max_chars > 0:
                text = f.read(max_chars)
            else:
                text = f.read()

        # Tokenize the entire text
        print(f"Tokenizing {len(text)/1e6:.1f}MB text...")
        self.tokens = self.enc.encode(text)
        self.num_samples = max(0, len(self.tokens) - block_size)

        print(f"TextDataset: {len(self.tokens):,} tokens → {self.num_samples:,} samples")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class StreamingDataset(IterableDataset):
    """Iterable-style dataset for very large text files.

    Reads on-the-fly without loading the entire file into memory.
    Useful for training on datasets larger than available RAM.
    """

    def __init__(
        self,
        file_path: str,
        block_size: int,
        encoding_name: str = "gpt2",
        buffer_size: int = 1_000_000,  # chars to load per chunk
        stride: int = 128,             # overlap between chunks
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.block_size = block_size
        self.encoding_name = encoding_name
        self.buffer_size = buffer_size
        self.stride = stride

    def __iter__(self):
        enc = tiktoken.get_encoding(self.encoding_name)
        buffer = ""

        with open(self.file_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(self.buffer_size)
                if not chunk:
                    break

                buffer += chunk
                if len(buffer) < self.block_size + 1:
                    continue

                tokens = enc.encode(buffer)

                # Yield sliding windows
                for i in range(0, len(tokens) - self.block_size, self.stride):
                    x = torch.tensor(tokens[i : i + self.block_size], dtype=torch.long)
                    y = torch.tensor(tokens[i + 1 : i + self.block_size + 1], dtype=torch.long)
                    yield x, y

                # Keep tail for next buffer
                tail = buffer[-self.block_size:] if len(buffer) > self.block_size else buffer
                buffer = tail


def create_dataloaders(
    train_file: str,
    val_file: str,
    block_size: int,
    batch_size: int,
    encoding_name: str = "gpt2",
    num_workers: int = 2,
    max_chars: int = 0,
    pin_memory: bool = False,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = TextDataset(train_file, block_size, encoding_name, max_chars=max_chars)
    val_dataset = TextDataset(val_file, block_size, encoding_name, max_chars=max(max_chars // 10, 100_000))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return train_loader, val_loader

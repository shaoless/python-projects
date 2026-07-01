"""Tests for dataset utilities: TextDataset, StreamingDataset, create_dataloaders.

Covers:
    - TextDataset initializes and returns correct (x, y) shapes
    - TextDataset y is x shifted by 1 (next-token prediction)
    - TextDataset computes number of samples correctly
    - TextDataset max_chars parameter limits text read
    - StreamingDataset iterates and yields correct (x, y) pairs
    - create_dataloaders returns DataLoader instances with correct batch shapes
    - Edge cases: very small files, empty files, max_chars=0
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.dataset import TextDataset, StreamingDataset, create_dataloaders


# ===================================================================
# TextDataset
# ===================================================================

class TestTextDataset:
    """Unit tests for the map-style TextDataset."""

    def test_init(self, dummy_text_file, tiny_config):
        """TextDataset initializes and is an instance of Dataset."""
        ds = TextDataset(dummy_text_file, tiny_config.block_size)
        assert isinstance(ds, Dataset)
        assert isinstance(ds, TextDataset)

    def test_shapes(self, dummy_text_file, tiny_config):
        """Each item returns (x, y) tensors of shape [block_size] with dtype long."""
        ds = TextDataset(dummy_text_file, tiny_config.block_size)
        x, y = ds[0]
        assert x.shape == (tiny_config.block_size,)
        assert y.shape == (tiny_config.block_size,)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_y_is_x_shifted_by_one(self, dummy_text_file, tiny_config):
        """y[t] == x[t+1] for all t in [0, block_size-1) (next-token prediction)."""
        ds = TextDataset(dummy_text_file, tiny_config.block_size)
        x, y = ds[0]
        assert torch.equal(x[1:], y[:-1]), (
            "y should be x shifted by 1 token (next-token prediction)"
        )

    def test_len_positive(self, dummy_text_file, tiny_config):
        """len(dataset) = max(0, total_tokens - block_size) > 0 for sufficiently large files."""
        ds = TextDataset(dummy_text_file, tiny_config.block_size)
        n = len(ds)
        assert n > 0, "A 3000-char file should produce at least 1 sample with block_size=32"

    def test_len_consistency(self, dummy_text_file, tiny_config):
        """Every index from 0 to len-1 is valid (no IndexError)."""
        ds = TextDataset(dummy_text_file, tiny_config.block_size)
        n = len(ds)
        for i in range(n):
            x, y = ds[i]
            assert x.shape == (tiny_config.block_size,)
            assert y.shape == (tiny_config.block_size,)
        # Also verify the last index + block_size doesn't exceed token list
        assert n <= len(ds.tokens) - tiny_config.block_size

    def test_max_chars_limits_data(self, dummy_text_file, tiny_config):
        """max_chars > 0 restricts the amount of text read, producing fewer samples."""
        ds_full = TextDataset(dummy_text_file, tiny_config.block_size, max_chars=0)
        ds_limited = TextDataset(
            dummy_text_file, tiny_config.block_size, max_chars=500
        )
        assert len(ds_limited) < len(ds_full), (
            f"max_chars=500 should give fewer samples than full file "
            f"({len(ds_limited)} < {len(ds_full)})"
        )

    def test_max_chars_zero_loads_all(self, dummy_text_file, tiny_config):
        """max_chars=0 reads the entire file, producing many samples."""
        ds = TextDataset(dummy_text_file, tiny_config.block_size, max_chars=0)
        assert len(ds) > 0, "Full file should produce samples"

    def test_very_small_file(self, tmp_path, tiny_config):
        """A file too small to fill block_size+1 tokens produces 0 samples."""
        path = tmp_path / "tiny.txt"
        path.write_text("Hello world", encoding="utf-8")
        ds = TextDataset(str(path), tiny_config.block_size)
        assert len(ds) == 0, "File with ~3 tokens should yield 0 samples at block_size=32"

    def test_empty_file(self, tmp_path, tiny_config):
        """An empty file produces 0 samples."""
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        ds = TextDataset(str(path), tiny_config.block_size)
        assert len(ds) == 0

    def test_encoding_name_parameter(self, dummy_text_file, tiny_config):
        """Different encoding names can be used (cl100k_base for GPT-4)."""
        ds = TextDataset(
            dummy_text_file, tiny_config.block_size, encoding_name="cl100k_base"
        )
        assert len(ds) > 0

    def test_y_boundary_no_overflow(self, dummy_text_file, tiny_config):
        """The last sample's y tensor does not reference beyond token list."""
        ds = TextDataset(dummy_text_file, tiny_config.block_size)
        n = len(ds)
        # Last sample should work without IndexError
        x, y = ds[n - 1]
        assert x.shape == (tiny_config.block_size,)
        assert y.shape == (tiny_config.block_size,)
        # Verify y's last token is the very next token after x's last token
        # y[-1] should be tokens[n - 1 + block_size + 1 - 1] = tokens[n + block_size - 1]
        # x[-1] should be tokens[n - 1 + block_size - 1] = tokens[n + block_size - 2]
        # They should be adjacent in the token stream
        last_x_token = ds.tokens[n - 1 + tiny_config.block_size - 1]
        last_y_token = ds.tokens[n - 1 + tiny_config.block_size]
        assert y[-1].item() == last_y_token
        assert x[-1].item() == last_x_token


# ===================================================================
# StreamingDataset
# ===================================================================

class TestStreamingDataset:
    """Unit tests for the iterable StreamingDataset."""

    def test_is_iterable_dataset(self, dummy_text_file, tiny_config):
        """StreamingDataset is an instance of IterableDataset."""
        ds = StreamingDataset(dummy_text_file, tiny_config.block_size)
        assert isinstance(ds, IterableDataset)

    def test_yields_x_y_pairs(self, dummy_text_file, tiny_config):
        """Iterating yields (x, y) tensors of shape [block_size] with dtype long."""
        ds = StreamingDataset(dummy_text_file, tiny_config.block_size)
        count = 0
        for x, y in ds:
            assert x.shape == (tiny_config.block_size,)
            assert y.shape == (tiny_config.block_size,)
            assert x.dtype == torch.long
            assert y.dtype == torch.long
            count += 1
            if count >= 5:
                break
        assert count > 0, "StreamingDataset should yield at least one sample"

    def test_y_is_x_shifted_by_one(self, dummy_text_file, tiny_config):
        """Each yielded pair satisfies y[:-1] == x[1:]."""
        ds = StreamingDataset(dummy_text_file, tiny_config.block_size)
        for x, y in ds:
            assert torch.equal(x[1:], y[:-1]), (
                "StreamingDataset y should be x shifted by 1"
            )
            break  # Check just the first sample

    def test_multiple_samples_have_different_content(self, dummy_text_file, tiny_config):
        """Consecutive samples are different (not all the same token sequence)."""
        ds = StreamingDataset(dummy_text_file, tiny_config.block_size)
        samples = []
        for x, _ in ds:
            samples.append(x.clone())
            if len(samples) >= 3:
                break
        # At least some samples should differ
        identical = all(torch.equal(samples[0], s) for s in samples[1:])
        assert not identical, "Consecutive samples should not all be identical"

    def test_very_small_file(self, tmp_path, tiny_config):
        """A file smaller than block_size yields no samples."""
        path = tmp_path / "tiny.txt"
        path.write_text("Hi", encoding="utf-8")
        ds = StreamingDataset(str(path), tiny_config.block_size)
        samples = list(ds)
        assert len(samples) == 0

    def test_empty_file(self, tmp_path, tiny_config):
        """An empty file yields no samples."""
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        ds = StreamingDataset(str(path), tiny_config.block_size)
        samples = list(ds)
        assert len(samples) == 0

    def test_produces_samples(self, dummy_text_file, tiny_config):
        """Full dummy file produces at least some samples in streaming mode."""
        ds = StreamingDataset(dummy_text_file, tiny_config.block_size)
        samples = list(ds)
        assert len(samples) >= 1, "StreamingDataset should produce at least 1 sample"

    def test_custom_stride(self, dummy_text_file, tiny_config):
        """Smaller stride produces more samples than larger stride."""
        ds_small_stride = StreamingDataset(
            dummy_text_file, tiny_config.block_size, stride=8
        )
        ds_large_stride = StreamingDataset(
            dummy_text_file, tiny_config.block_size, stride=512
        )
        samples_small = sum(1 for _ in ds_small_stride)
        samples_large = sum(1 for _ in ds_large_stride)
        assert samples_small >= samples_large, (
            f"Smaller stride ({8}) should produce >= samples than large stride ({512})"
        )


# ===================================================================
# create_dataloaders
# ===================================================================

class TestCreateDataLoaders:
    """Tests for the create_dataloaders helper."""

    def test_returns_dataloaders(self, dummy_text_file, dummy_val_file, tiny_config):
        """create_dataloaders returns (DataLoader, DataLoader)."""
        train_loader, val_loader = create_dataloaders(
            train_file=dummy_text_file,
            val_file=dummy_val_file,
            block_size=tiny_config.block_size,
            batch_size=2,
            num_workers=0,
        )
        assert isinstance(train_loader, DataLoader), "train_loader should be a DataLoader"
        assert isinstance(val_loader, DataLoader), "val_loader should be a DataLoader"

    def test_batch_shapes(self, dummy_text_file, dummy_val_file, tiny_config):
        """Batches from loaders have shape (batch_size, block_size)."""
        train_loader, val_loader = create_dataloaders(
            train_file=dummy_text_file,
            val_file=dummy_val_file,
            block_size=tiny_config.block_size,
            batch_size=2,
            num_workers=0,
        )
        x, y = next(iter(train_loader))
        assert x.shape == (2, tiny_config.block_size), (
            f"Expected (2, {tiny_config.block_size}), got {x.shape}"
        )
        assert y.shape == (2, tiny_config.block_size)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_val_loader_batches(self, dummy_text_file, dummy_val_file, tiny_config):
        """Validation loader also yields correct shapes."""
        _, val_loader = create_dataloaders(
            train_file=dummy_text_file,
            val_file=dummy_val_file,
            block_size=tiny_config.block_size,
            batch_size=2,
            num_workers=0,
        )
        x, y = next(iter(val_loader))
        assert x.shape == (2, tiny_config.block_size)
        assert y.shape == (2, tiny_config.block_size)

    def test_drop_last_removes_incomplete_batch(self, dummy_text_file, dummy_val_file, tiny_config):
        """drop_last=True removes the final incomplete batch (dataset might have remainder)."""
        train_loader, _ = create_dataloaders(
            train_file=dummy_text_file,
            val_file=dummy_val_file,
            block_size=tiny_config.block_size,
            batch_size=2,
            num_workers=0,
        )
        # Every batch should have exactly batch_size samples
        for x, y in train_loader:
            assert x.shape[0] == 2

    def test_custom_encoding_name(self, dummy_text_file, dummy_val_file, tiny_config):
        """create_dataloaders works with different tokenizer encodings."""
        train_loader, _ = create_dataloaders(
            train_file=dummy_text_file,
            val_file=dummy_val_file,
            block_size=tiny_config.block_size,
            batch_size=2,
            encoding_name="cl100k_base",
            num_workers=0,
        )
        x, y = next(iter(train_loader))
        assert x.shape == (2, tiny_config.block_size)

    def test_files_too_small_for_shuffle_raises(self, tmp_path, tiny_config):
        """Files producing 0 samples raise ValueError because shuffle=True causes
        RandomSampler(num_samples=0) to fail.
        This is an inherent limitation of ``shuffle=True`` with zero-length datasets."""
        train_path = tmp_path / "small_train.txt"
        val_path = tmp_path / "small_val.txt"
        train_path.write_text("Hello", encoding="utf-8")
        val_path.write_text("World", encoding="utf-8")
        with pytest.raises(ValueError, match="num_samples should be a positive integer"):
            create_dataloaders(
                train_file=str(train_path),
                val_file=str(val_path),
                block_size=tiny_config.block_size,
                batch_size=2,
                num_workers=0,
            )

    def test_barely_sufficient_file(self, tmp_path, tiny_config):
        """A file that is barely large enough produces dataloaders with at least one batch."""
        train_path = tmp_path / "barely_train.txt"
        val_path = tmp_path / "barely_val.txt"
        # Need enough chars to produce >= block_size+1 tokens.
        # "hello world " repeated enough times should be at least 33 tokens
        # with GPT-2 tokenizer (~1 token per word + space).
        train_path.write_text("hello world " * 500, encoding="utf-8")
        val_path.write_text("hello world " * 500, encoding="utf-8")
        train_loader, val_loader = create_dataloaders(
            train_file=str(train_path),
            val_file=str(val_path),
            block_size=tiny_config.block_size,
            batch_size=2,
            num_workers=0,
        )
        assert len(train_loader) > 0
        assert len(val_loader) > 0

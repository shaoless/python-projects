"""Tests for tokenizer utilities: get_pretrained_tokenizer, encode/decode roundtrip,
prepare_text_for_training, and other helper functions.

Covers:
    - get_pretrained_tokenizer returns tiktoken.Encoding for common names
    - Encode/decode roundtrip preserves text (ASCII, Unicode, empty, long)
    - prepare_text_for_training filters short lines and concatenates files
    - train_tokenizer raises NotImplementedError as documented
    - Edge cases: empty input files, all lines filtered, empty string
"""

import os

import pytest
import tiktoken

from src.tokenizer import (
    get_pretrained_tokenizer,
    prepare_text_for_training,
    train_tokenizer,
    train_hf_tokenizer,
    load_hf_tokenizer,
)


# ===================================================================
# get_pretrained_tokenizer
# ===================================================================

class TestGetPretrainedTokenizer:
    """Tests for get_pretrained_tokenizer()."""

    def test_gpt2_returns_tiktoken_encoding(self):
        """get_pretrained_tokenizer('gpt2') returns a tiktoken.Encoding."""
        enc = get_pretrained_tokenizer("gpt2")
        assert isinstance(enc, tiktoken.Encoding)
        assert enc.name == "gpt2"

    def test_default_is_gpt2(self):
        """Default argument is 'gpt2'."""
        enc = get_pretrained_tokenizer()
        assert enc.name == "gpt2"

    @pytest.mark.parametrize("name", ["gpt2", "cl100k_base"])
    def test_various_encoding_names(self, name):
        """Various encoding names return valid tiktoken.Encoding objects.
        Note: o200k_base is excluded — it downloads a large file that can fail
        on restricted networks.
        """
        enc = get_pretrained_tokenizer(name)
        assert isinstance(enc, tiktoken.Encoding)
        assert enc.name == name

    def test_gpt2_vocab_size(self):
        """GPT-2 tokenizer has 50257 tokens in the vocabulary."""
        enc = get_pretrained_tokenizer("gpt2")
        # n_vocab includes the base vocabulary; special tokens may add more
        assert enc.n_vocab >= 50257


# ===================================================================
# Encode / Decode roundtrip
# ===================================================================

class TestEncodeDecodeRoundtrip:
    """Tests that tokenizer.encode -> tokenizer.decode preserves text."""

    def test_roundtrip_ascii(self):
        """Simple ASCII sentence round-trips correctly."""
        enc = get_pretrained_tokenizer("gpt2")
        text = "The quick brown fox jumps over the lazy dog."
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        assert decoded == text, f"Roundtrip failed: '{text}' -> '{decoded}'"

    def test_roundtrip_with_punctuation(self):
        """Text with punctuation round-trips correctly."""
        enc = get_pretrained_tokenizer("gpt2")
        text = "Hello, world! How are you? I'm fine: 1, 2, 3..."
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        assert decoded == text

    def test_roundtrip_with_numbers(self):
        """Text containing numbers round-trips correctly."""
        enc = get_pretrained_tokenizer("gpt2")
        text = "Test 123 numbers 4567 and 89."
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        assert decoded == text

    def test_roundtrip_empty_string(self):
        """Empty string encodes to [] and decodes to ''."""
        enc = get_pretrained_tokenizer("gpt2")
        tokens = enc.encode("")
        assert tokens == [], "Empty string should encode to empty list"
        decoded = enc.decode(tokens)
        assert decoded == ""

    def test_roundtrip_very_long_string(self):
        """A long repeated string round-trips correctly."""
        enc = get_pretrained_tokenizer("gpt2")
        text = "hello world " * 500
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        assert len(tokens) > 0, "Long string should produce tokens"
        assert decoded == text, "Long string roundtrip should preserve text"

    def test_roundtrip_multiple_calls(self):
        """Multiple encode/decode calls are consistent (tokenizer state is unchanged)."""
        enc = get_pretrained_tokenizer("gpt2")
        texts = [
            "First sentence.",
            "Second sentence with different content.",
            "Third: numbers 42 and symbols!",
        ]
        for text in texts:
            tokens = enc.encode(text)
            decoded = enc.decode(tokens)
            assert decoded == text

    def test_encode_returns_integers(self):
        """encode returns a list of integers (token IDs)."""
        enc = get_pretrained_tokenizer("gpt2")
        tokens = enc.encode("Hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        for t in tokens:
            assert isinstance(t, int)
            assert t >= 0

    def test_decode_with_invalid_token_raises(self):
        """decode with an out-of-range token ID raises KeyError (tiktoken validation)."""
        enc = get_pretrained_tokenizer("gpt2")
        with pytest.raises(KeyError, match="Invalid token"):
            enc.decode([0, 99999, 1])


# ===================================================================
# prepare_text_for_training
# ===================================================================

class TestPrepareTextForTraining:
    """Tests for prepare_text_for_training()."""

    def test_filters_short_lines(self, multi_line_text_file, tmp_path):
        """Lines shorter than min_length are filtered out of the output."""
        output = str(tmp_path / "output.txt")
        prepare_text_for_training(
            [multi_line_text_file], output, min_length=50
        )
        with open(output, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # All output lines should be >= 50 chars
        for line in lines:
            assert len(line) >= 50, f"Line too short: '{line}' (len={len(line)})"

        # The fixture has 5 lines; 2 are short ("Short." and "Tiny.")
        assert len(lines) == 3, f"Expected 3 lines >=50 chars, got {len(lines)}"

    def test_output_file_created(self, multi_line_text_file, tmp_path):
        """Output file is created and contains content."""
        output = str(tmp_path / "output.txt")
        prepare_text_for_training([multi_line_text_file], output, min_length=10)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_empty_input_list(self, tmp_path):
        """Empty input_files list produces an empty (0-byte) output file."""
        output = str(tmp_path / "empty_output.txt")
        prepare_text_for_training([], output, min_length=10)
        assert os.path.exists(output)
        # File should exist but be empty
        assert os.path.getsize(output) == 0

    def test_all_lines_filtered(self, tmp_path):
        """When all lines are shorter than min_length, output is empty."""
        path = tmp_path / "all_short.txt"
        path.write_text("Hi.\nBye.\nOK.\n", encoding="utf-8")
        output = str(tmp_path / "filtered.txt")
        prepare_text_for_training([str(path)], output, min_length=100)
        assert os.path.exists(output)
        assert os.path.getsize(output) == 0

    def test_multiple_input_files(self, multi_line_text_file, tmp_path):
        """Multiple input files are concatenated into the output."""
        # Create a second file with one long line
        second_file = tmp_path / "second.txt"
        second_file.write_text(
            "This is an additional long line for concatenation testing purposes.\n",
            encoding="utf-8",
        )
        output = str(tmp_path / "combined.txt")
        prepare_text_for_training(
            [multi_line_text_file, str(second_file)], output, min_length=50
        )
        with open(output, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        # 3 lines from multi_line_text_file + 1 from second_file = 4
        assert len(lines) == 4

    def test_min_length_zero_keeps_all_lines(self, multi_line_text_file, tmp_path):
        """min_length=0 keeps all non-empty lines."""
        output = str(tmp_path / "all_lines.txt")
        prepare_text_for_training([multi_line_text_file], output, min_length=0)
        with open(output, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 5, "All 5 lines should be kept with min_length=0"

    def test_lines_are_stripped(self, tmp_path):
        """Whitespace is stripped from each line before length check and writing."""
        path = tmp_path / "whitespace.txt"
        # Line has leading whitespace but enough content after stripping
        path.write_text(
            "   \n"
            "   Short.\n"
            "   This line has leading spaces but is long enough after stripping.\n",
            encoding="utf-8",
        )
        output = str(tmp_path / "stripped.txt")
        prepare_text_for_training([str(path)], output, min_length=50)
        with open(output, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        # Only the long line should survive (not the blank line or the short one)
        assert len(lines) == 1
        # The written line should not have leading spaces (it was stripped)
        assert not lines[0].startswith("   ")


# ===================================================================
# train_tokenizer
# ===================================================================

class TestTrainTokenizer:
    """Tests for train_tokenizer().

    Note: train_tokenizer is documented as not supported in tiktoken.
    It raises an error when called.
    """

    def test_raises_error(self, tmp_path):
        """train_tokenizer raises an error (tiktoken does not support training from scratch)."""
        path = tmp_path / "dummy.txt"
        path.write_text("some training text", encoding="utf-8")
        with pytest.raises(Exception):
            train_tokenizer(str(path))


# ===================================================================
# train_hf_tokenizer and load_hf_tokenizer (smoke tests)
# ===================================================================

class TestHFTokenizerFunctions:
    """Smoke tests for HuggingFace tokenizer functions (no file I/O)."""

    def test_train_hf_tokenizer_is_callable(self):
        """train_hf_tokenizer is a callable function."""
        assert callable(train_hf_tokenizer)

    def test_load_hf_tokenizer_is_callable(self):
        """load_hf_tokenizer is a callable function."""
        assert callable(load_hf_tokenizer)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline import source_encoding as se
from pipeline import compression as lz
from pipeline import decoding


def _forward_pipeline(text: str):
    """Mimic the sender side: huffman-encode -> pack bits into bytes -> lz77-compress."""
    bitstring, codes = se.huffman_encode(text)
    # pad to byte boundary
    pad = (-len(bitstring)) % 8
    padded_bits = bitstring + "0" * pad
    byte_data = bytes(int(padded_bits[i:i + 8], 2) for i in range(0, len(padded_bits), 8))
    tokens = lz.lz77_compress(byte_data)
    return tokens, codes


def test_decode_full_round_trip():
    text = "the quick brown fox jumps over the lazy dog"
    tokens, codes = _forward_pipeline(text)
    result = decoding.decode_full(tokens, codes, text)
    assert result["success"] is True
    assert result["recovered"] == text
    assert result["diff_positions"] == []


def test_decode_full_detects_mismatch():
    text = "hello world"
    tokens, codes = _forward_pipeline(text)
    result = decoding.decode_full(tokens, codes, "hello earth")
    assert result["success"] is False
    assert len(result["diff_positions"]) > 0


def test_decode_binary_only():
    from pipeline import binary_repr as br
    bits = br.text_to_binary("test")
    assert decoding.decode_binary_only(bits) == "test"

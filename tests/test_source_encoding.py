import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import math
from pipeline import source_encoding as se
from pipeline import binary_repr as br


def test_frequency_table():
    freq = se.build_frequency_table("aabbbc")
    assert freq == {"a": 2, "b": 3, "c": 1}


def test_codes_are_prefix_free():
    text = "this is an example of a huffman tree"
    freq = se.build_frequency_table(text)
    tree = se.build_huffman_tree(freq)
    codes = se.generate_codes(tree)
    items = list(codes.values())
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i != j:
                assert not b.startswith(a), f"{a} is a prefix of {b}"


def test_encode_decode_round_trip():
    texts = [
        "hello world",
        "aaaaaaaaaa",
        "the quick brown fox jumps over the lazy dog",
        "a",
        "ab",
    ]
    for text in texts:
        bits, codes = se.huffman_encode(text)
        decoded = se.huffman_decode(bits, codes)
        assert decoded == text, f"round trip failed for {text!r}"


def test_single_character_repeated_gets_valid_code():
    bits, codes = se.huffman_encode("zzzz")
    assert codes["z"] == "0"
    assert bits == "0000"
    assert se.huffman_decode(bits, codes) == "zzzz"


def test_empty_string():
    bits, codes = se.huffman_encode("")
    assert bits == ""
    assert codes == {}


def test_entropy_matches_manual_calculation():
    # "aab" -> p(a)=2/3, p(b)=1/3
    freq = {"a": 2, "b": 1}
    h = se.shannon_entropy(freq)
    expected = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    assert abs(h - expected) < 1e-9


def test_avg_code_length_at_least_entropy():
    text = "the quick brown fox jumps over the lazy dog repeatedly for entropy testing"
    freq = se.build_frequency_table(text)
    bits, codes = se.huffman_encode(text)
    h = se.shannon_entropy(freq)
    avg_len = se.average_code_length(freq, codes)
    # Shannon's source coding theorem: H(X) <= L < H(X) + 1
    assert h - 1e-9 <= avg_len < h + 1


def test_binary_round_trip():
    for text in ["hello", "Data 123!", "π approx", ""]:
        bits = br.text_to_binary(text)
        assert len(bits) % 8 == 0
        assert br.binary_to_text(bits) == text


def test_visualize_bits_structure():
    bits = br.text_to_binary("Hi")
    grid = br.visualize_bits(bits)
    assert len(grid) == 2
    assert grid[0]["ascii"] == "H"
    assert grid[1]["ascii"] == "i"
    assert grid[0]["value"] == ord("H")


def test_bytes_to_bits_round_trip():
    import random
    random.seed(0)
    for n in [0, 1, 16, 100]:
        data = bytes(random.randint(0, 255) for _ in range(n))
        bits = br.bytes_to_bits(data)
        assert len(bits) == n * 8
        assert br.bits_to_bytes(bits) == data

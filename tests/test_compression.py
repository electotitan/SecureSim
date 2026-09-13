import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline import compression as lz


def test_round_trip_simple():
    data = b"abababababab"
    tokens = lz.lz77_compress(data)
    assert lz.lz77_decompress(tokens) == data


def test_round_trip_repeated_single_char_overlapping_match():
    data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    tokens = lz.lz77_compress(data)
    assert lz.lz77_decompress(tokens) == data


def test_round_trip_no_repetition():
    data = b"xyz"
    tokens = lz.lz77_compress(data)
    assert lz.lz77_decompress(tokens) == data
    # No repetition -> every token should be a literal (length 0)
    assert all(length == 0 for _, length, _ in tokens)


def test_round_trip_empty():
    assert lz.lz77_decompress(lz.lz77_compress(b"")) == b""


def test_round_trip_english_text():
    data = b"the quick brown fox jumps over the lazy dog. the quick brown fox runs away."
    tokens = lz.lz77_compress(data)
    assert lz.lz77_decompress(tokens) == data


def test_round_trip_binary_data():
    import random
    random.seed(1)
    data = bytes(random.randint(0, 255) for _ in range(500))
    tokens = lz.lz77_compress(data)
    assert lz.lz77_decompress(tokens) == data


def test_compression_report_ratio_for_repetitive_data():
    data = b"a" * 1000
    report = lz.compression_report(data)
    assert report["original_size_bytes"] == 1000
    assert report["compression_ratio"] > 1  # highly repetitive data should compress well


def test_window_size_bounds_match_distance():
    data = b"x" * 50 + b"ABCDEFGHIJKLMNOPQR" + b"y" * 50 + b"ABCDEFGHIJKLMNOPQR"
    tokens = lz.lz77_compress(data, window_size=10, lookahead_size=18)
    for offset, length, _ in tokens:
        assert offset <= 10
    assert lz.lz77_decompress(tokens) == data

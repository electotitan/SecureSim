"""Checkpoint 4: LZ77 compression tests."""
import random

import pytest

from backend.pipeline.compression import (
    _longest_match_fast, _longest_match_reference,
    DEFAULT_LOOKAHEAD_SIZE, DEFAULT_WINDOW_SIZE, MAX_LOOKAHEAD_SIZE, MAX_WINDOW_SIZE,
    compression_stats, describe_tokens, deserialize_tokens, lz77_compress,
    lz77_compress_reference, lz77_decompress, serialize_tokens, token_bit_cost,
)

ENGINES = [lz77_compress, lz77_compress_reference]


def rand_bytes(rng, n, alphabet):
    return bytes(rng.randrange(alphabet) for _ in range(n))


# ------------------------------------------------------------------ independent checker
def independent_longest_match(data, i, window_size, lookahead_size):
    """Written separately from the library code: (length, offset) under the documented rules."""
    n = len(data)
    remaining = n - i
    max_len = remaining if remaining <= lookahead_size else lookahead_size - 1
    best = (0, 0)
    for offset in range(1, min(window_size, i) + 1):
        k = 0
        while k < max_len and data[i - offset + k] == data[i + k]:
            k += 1
        if k > best[0]:
            best = (k, offset)
    return best


def check_tokens_follow_rules(data, tokens, window_size, lookahead_size):
    pos = 0
    for idx, (offset, length, nxt) in enumerate(tokens):
        exp_len, exp_off = independent_longest_match(data, pos, window_size, lookahead_size)
        assert length == exp_len, f"token {idx} at {pos}: length {length} != longest {exp_len}"
        assert offset == (exp_off if exp_len else 0), f"token {idx}: offset {offset} != smallest {exp_off}"
        assert length + len(nxt) <= lookahead_size
        assert offset <= min(window_size, pos)
        if not nxt:
            assert idx == len(tokens) - 1 and pos + length == len(data)
        pos += length + len(nxt)
    assert pos == len(data)


# ------------------------------------------------------------------ basic behaviour
@pytest.mark.parametrize("engine", ENGINES)
def test_empty_data(engine):
    assert engine(b"") == []
    assert lz77_decompress([]) == b""


@pytest.mark.parametrize("engine", ENGINES)
def test_single_byte_is_a_literal(engine):
    assert engine(b"a") == [(0, 0, b"a")]


@pytest.mark.parametrize("engine", ENGINES)
def test_repeated_bytes_use_overlapping_match_and_end_token(engine):
    # 'a' literal, then (offset 1, length 3) reaching end of input: empty next byte.
    assert engine(b"aaaa") == [(0, 0, b"a"), (1, 3, b"")]


@pytest.mark.parametrize("engine", ENGINES)
def test_abracadabra_known_tokens(engine):
    # Derived by hand (indices: a0 b1 r2 a3 c4 a5 d6 a7 b8 r9 a10):
    #  a, b, r literals; i=3 'a' matches idx0 (offset 3) next 'c'; i=5 'a' nearest earlier 'a'
    #  is idx3 (offset 2) next 'd'; i=7 "abra" matches idx0 (offset 7) and reaches end of input.
    assert engine(b"abracadabra") == [
        (0, 0, b"a"), (0, 0, b"b"), (0, 0, b"r"), (3, 1, b"c"), (2, 1, b"d"), (7, 4, b""),
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_non_repeating_data_is_all_literals(engine):
    data = bytes(range(50))
    tokens = engine(data)
    assert tokens == [(0, 0, bytes([b])) for b in data]


def test_overlapping_decompression():
    assert lz77_decompress([(0, 0, b"a"), (1, 5, b"b")]) == b"aaaaaab"
    assert lz77_decompress([(0, 0, b"a"), (0, 0, b"b"), (2, 6, b"")]) == b"abababab"
    # offset 2, length 5 over "ab" copies a b a b a (each copied byte feeds the next)
    assert lz77_decompress([(0, 0, b"a"), (0, 0, b"b"), (2, 5, b"c")]) == b"abababac"


def test_all_byte_values_round_trip():
    data = bytes(range(256)) * 3
    assert lz77_decompress(lz77_compress(data)) == data


def test_unicode_text_via_utf8():
    text = "héllo wörld 😀 日本語 " * 20
    data = text.encode("utf-8")
    assert lz77_decompress(lz77_compress(data)).decode("utf-8") == text


# ------------------------------------------------------------------ window and lookahead
def test_window_limits_how_far_back_matches_reach():
    data = b"abc" + b"xyzxyzxyz" + b"abc"
    big = lz77_compress(data, window_size=64, lookahead_size=18)
    small = lz77_compress(data, window_size=4, lookahead_size=18)
    assert max(t[0] for t in big) > 4          # the far 'abc' is referenced with a large window
    assert max(t[0] for t in small) <= 4       # ... and impossible with window 4
    assert lz77_decompress(big) == data == lz77_decompress(small)


def test_lookahead_limits_token_coverage():
    data = b"a" * 100
    for look in (1, 2, 5, 18, 40):
        tokens = lz77_compress(data, lookahead_size=look)
        assert all(length + len(nxt) <= look for _, length, nxt in tokens)
        assert lz77_decompress(tokens) == data
    # lookahead 1 can never copy more than a single end-of-input byte
    assert lz77_compress(b"aa", lookahead_size=1) == [(0, 0, b"a"), (1, 1, b"")]


def test_end_of_input_match_can_use_whole_lookahead():
    # remaining == lookahead_size and fully matching -> single end-of-input token of length 18
    data = b"a" + b"a" * 18
    tokens = lz77_compress(data, lookahead_size=18)
    assert tokens == [(0, 0, b"a"), (1, 18, b"")]


def test_tokens_cover_each_byte_once_and_respect_limits():
    rng = random.Random(3)
    for _ in range(60):
        data = rand_bytes(rng, rng.randint(1, 200), rng.choice([1, 2, 3, 8]))
        w, l = rng.randint(1, 40), rng.randint(1, 20)
        tokens = lz77_compress(data, w, l)
        check_tokens_follow_rules(data, tokens, w, l)


# ------------------------------------------------------------------ engines agree and are optimal
def test_fast_engine_matches_reference_engine_exactly():
    rng = random.Random(2024)
    for _ in range(400):
        data = rand_bytes(rng, rng.randint(0, 300), rng.choice([1, 2, 3, 4, 16, 256]))
        w, l = rng.randint(1, 64), rng.randint(1, 25)
        assert lz77_compress(data, w, l) == lz77_compress_reference(data, w, l), (data, w, l)


def test_engines_agree_with_default_parameters_on_larger_inputs():
    rng = random.Random(77)
    for alphabet in (1, 2, 4, 256):
        data = rand_bytes(rng, 5000, alphabet)
        assert lz77_compress(data) == lz77_compress_reference(data)


def test_longest_match_and_smallest_offset_against_independent_checker():
    rng = random.Random(8)
    for _ in range(150):
        data = rand_bytes(rng, rng.randint(1, 120), rng.choice([1, 2, 3]))
        w, l = rng.randint(1, 30), rng.randint(1, 12)
        for engine in ENGINES:
            check_tokens_follow_rules(data, engine(data, w, l), w, l)


def test_tie_break_prefers_smallest_offset():
    # At position 6, "ab" occurs at offsets 6 (idx0) and 3 (idx3); the closer one must win.
    data = b"abxabyab"
    tokens = lz77_compress(data, 100, 18)
    assert tokens[-1] == (3, 2, b"")                    # offset 3 (idx 3), not offset 6 (idx 0)


# ------------------------------------------------------------------ round trips
@pytest.mark.parametrize("engine", ENGINES)
def test_round_trip_random(engine):
    rng = random.Random(42)
    for _ in range(300):
        data = rand_bytes(rng, rng.randint(0, 250), rng.choice([1, 2, 3, 5, 256]))
        w, l = rng.randint(1, 50), rng.randint(1, 24)
        assert lz77_decompress(engine(data, w, l)) == data


def test_round_trip_special_patterns():
    for data in (b"a" * 1000, b"ab" * 500, b"abc" * 333 + b"x", bytes(1000), b"\xff" * 77,
                 (b"hello world " * 40)):
        assert lz77_decompress(lz77_compress(data)) == data


def test_decompress_with_window_and_lookahead_checks():
    tokens = lz77_compress(b"abcabcabc", 8, 6)
    assert lz77_decompress(tokens, window_size=8, lookahead_size=6) == b"abcabcabc"
    with pytest.raises(ValueError, match="window_size"):
        lz77_decompress(tokens, window_size=2)
    with pytest.raises(ValueError, match="lookahead_size"):
        lz77_decompress(tokens, lookahead_size=2)


# ------------------------------------------------------------------ invalid tokens
@pytest.mark.parametrize("tokens, fragment", [
    ([(1, 1, b"a")], "before the start"),                       # nothing produced yet
    ([(0, 0, b"a"), (2, 1, b"b")], "before the start"),         # offset 2 > 1 byte of output
    ([(0, 0, b"a"), (0, 3, b"b")], "offset >= 1"),              # match without offset
    ([(0, 0, b"a"), (1, 0, b"b")], "offset 0"),                 # literal with offset
    ([(-1, 1, b"a")], "non-negative"),
    ([(0, 0, b"a"), (1, -1, b"b")], "non-negative"),
    ([(0, 0, b"ab")], "exactly one byte"),
    ([(0, 0, b"")], "no progress"),
    ([(0, 0, b"a"), (0, 0, b""), (0, 0, b"b")], "no progress"),
    ([(0, 0, b""), (0, 0, b"a")], "no progress"),
    ([(0, 0, b"a"), (1, 1, b""), (0, 0, b"b")], "only allowed on the final"),
    ([(0, 0, "a")], "must be bytes"),
    ([(0.0, 0, b"a")], "integers"),
    ([(True, 1, b"a")], "integers"),
    ([(0, 0)], "triple"),
    ([(0, 0, b"a", 1)], "triple"),
    (["abc"], "triple"),
    ([None], "triple"),
])
def test_invalid_tokens_rejected(tokens, fragment):
    with pytest.raises(ValueError, match=fragment):
        lz77_decompress(tokens)


def test_non_sequence_tokens_rejected():
    for bad in ("abc", 5, None, {"a": 1}, b"abc"):
        with pytest.raises(TypeError):
            lz77_decompress(bad)


def test_decompression_bomb_guard():
    with pytest.raises(ValueError, match="limit"):
        lz77_decompress([(0, 0, b"a"), (1, 10 ** 9, b"")])
    with pytest.raises(ValueError, match="limit"):
        lz77_decompress([(0, 0, b"a"), (1, 100, b"")], max_output=50)
    assert len(lz77_decompress([(0, 0, b"a"), (1, 99, b"")], max_output=100)) == 100


def test_list_tokens_are_accepted_like_tuples():
    # JSON round trips turn tuples into lists
    assert lz77_decompress([[0, 0, b"a"], [1, 3, b""]]) == b"aaaa"


# ------------------------------------------------------------------ parameter validation
@pytest.mark.parametrize("w, l", [(0, 18), (-1, 18), (4096, 0), (4096, -3),
                                  (MAX_WINDOW_SIZE + 1, 18), (4096, MAX_LOOKAHEAD_SIZE + 1)])
def test_invalid_parameters_rejected(w, l):
    for engine in ENGINES:
        with pytest.raises(ValueError):
            engine(b"abc", w, l)


@pytest.mark.parametrize("w, l", [(1.5, 18), ("4096", 18), (True, 18), (4096, None), (4096, 2.0)])
def test_non_int_parameters_rejected(w, l):
    with pytest.raises(TypeError):
        lz77_compress(b"abc", w, l)


@pytest.mark.parametrize("bad", ["abc", 123, None, [1, 2], 1.5])
def test_non_bytes_data_rejected(bad):
    for engine in ENGINES:
        with pytest.raises(TypeError):
            engine(bad)


def test_bytearray_accepted():
    assert lz77_compress(bytearray(b"aaaa")) == lz77_compress(b"aaaa")


def test_boundary_parameter_values_work():
    data = b"abcabcabc" * 5
    for w, l in [(1, 1), (1, 255), (32768, 1), (MAX_WINDOW_SIZE, MAX_LOOKAHEAD_SIZE)]:
        assert lz77_decompress(lz77_compress(data, w, l)) == data


# ------------------------------------------------------------------ serialisation and sizes
def test_serialize_empty_is_empty():
    assert serialize_tokens([]) == b""
    assert deserialize_tokens(b"") == []


def test_serialized_length_matches_documented_formula():
    rng = random.Random(5)
    for _ in range(100):
        data = rand_bytes(rng, rng.randint(1, 200), rng.choice([1, 2, 16]))
        w, l = rng.randint(1, 100), rng.randint(1, 30)
        tokens = lz77_compress(data, w, l)
        bits = sum(w.bit_length() + l.bit_length() + 1 + 8 * len(t[2]) for t in tokens)
        blob = serialize_tokens(tokens, w, l)
        assert len(blob) == 1 + (bits + 7) // 8
        assert blob[0] == (-bits) % 8
        stats = compression_stats(len(data), tokens, w, l)
        assert stats["compressed_bytes"] == len(blob)
        assert stats["payload_bits"] == bits


def test_serialize_round_trip():
    rng = random.Random(6)
    for _ in range(150):
        data = rand_bytes(rng, rng.randint(0, 250), rng.choice([1, 2, 4, 256]))
        w, l = rng.randint(1, 200), rng.randint(1, 40)
        tokens = lz77_compress(data, w, l)
        blob = serialize_tokens(tokens, w, l)
        assert deserialize_tokens(blob, w, l) == tokens
        assert lz77_decompress(deserialize_tokens(blob, w, l)) == data


def test_default_parameter_bit_widths():
    assert DEFAULT_WINDOW_SIZE.bit_length() == 13 and DEFAULT_LOOKAHEAD_SIZE.bit_length() == 5
    assert token_bit_cost((0, 0, b"a"), 4096, 18) == 13 + 5 + 1 + 8 == 27
    assert token_bit_cost((1, 3, b""), 4096, 18) == 13 + 5 + 1 == 19


def test_serialize_rejects_out_of_range_fields():
    with pytest.raises(ValueError):
        serialize_tokens([(0, 0, b"a"), (9, 1, b"b")], window_size=8, lookahead_size=18)
    with pytest.raises(ValueError):
        serialize_tokens([(0, 0, b"a"), (1, 19, b"b")], window_size=8, lookahead_size=18)


@pytest.mark.parametrize("blob", [
    bytes([8]) + b"\x00",          # padding count out of range (must be 0..7)
    bytes([0]),                    # header only, no token bits
    bytes([7]) + b"\x00",          # only 1 real bit: cannot hold a token
    bytes([0]) + b"\x00",          # 8 bits: still shorter than one 19-bit token
])
def test_deserialize_rejects_malformed(blob):
    with pytest.raises(ValueError):
        deserialize_tokens(blob, 4096, 18)


def test_deserialize_rejects_truncated_and_nonzero_padding():
    blob = serialize_tokens(lz77_compress(b"hello world"), 4096, 18)
    with pytest.raises(ValueError):
        deserialize_tokens(blob[:-1], 4096, 18)                      # chops a token
    tokens = [(0, 0, b"a"), (1, 3, b"")]
    good = serialize_tokens(tokens, 4096, 18)
    assert good[0] != 0                                               # has padding bits
    bad = good[:-1] + bytes([good[-1] | 1])                            # set a padding bit
    with pytest.raises(ValueError, match="padding bits"):
        deserialize_tokens(bad, 4096, 18)


def test_deserialize_rejects_fields_beyond_declared_sizes():
    # Serialise with a big window, then read back claiming a small one: offset field too large.
    blob = serialize_tokens([(0, 0, b"a"), (1, 1, b"b")], window_size=4096, lookahead_size=18)
    with pytest.raises(ValueError):
        deserialize_tokens(blob, window_size=1, lookahead_size=18)


# ------------------------------------------------------------------ statistics
def test_stats_known_values_expansion_example():
    tokens = lz77_compress(b"aaaa")                       # [(0,0,'a'), (1,3,b'')]
    s = compression_stats(4, tokens)
    assert s["token_count"] == 2 and s["literal_count"] == 1 and s["match_count"] == 1
    assert s["payload_bits"] == 27 + 19 == 46
    assert s["compressed_bytes"] == 1 + 6 == 7
    assert s["compression_ratio"] == pytest.approx(4 / 7)
    assert s["space_saved_percent"] == pytest.approx((1 - 7 / 4) * 100) == pytest.approx(-75.0)
    assert s["expanded"] is True                           # short data expands: honest reporting


def test_stats_repetitive_data_compresses():
    data = b"abc" * 1000
    s = compression_stats(len(data), lz77_compress(data))
    assert s["compressed_bytes"] < len(data) and s["expanded"] is False
    assert s["compression_ratio"] > 1 and 0 < s["space_saved_percent"] < 100
    assert s["compression_ratio"] == pytest.approx(len(data) / s["compressed_bytes"])


def test_stats_incompressible_data_expands():
    rng = random.Random(1)
    data = bytes(rng.randrange(256) for _ in range(2000))
    s = compression_stats(len(data), lz77_compress(data, 64, 18))   # small window: few repeats
    assert s["expanded"] is True and s["compression_ratio"] < 1 and s["space_saved_percent"] < 0


def test_stats_empty_input_has_undefined_ratio():
    s = compression_stats(0, [])
    assert s["compressed_bytes"] == 0 and s["compression_ratio"] is None
    assert s["space_saved_percent"] is None and s["expanded"] is False


def test_token_count_alone_does_not_determine_size():
    # Same number of tokens, different sizes (literals carry a byte; end-of-input matches do not).
    a = [(0, 0, b"a"), (0, 0, b"b")]
    b = [(0, 0, b"a"), (1, 1, b"")]
    assert len(a) == len(b)
    assert compression_stats(2, a)["compressed_bytes"] != compression_stats(2, b)["compressed_bytes"]


# ------------------------------------------------------------------ token descriptions
def test_describe_tokens():
    data = b"abracadabra"
    tokens = lz77_compress(data)
    desc = describe_tokens(data, tokens)
    assert [d["position"] for d in desc] == [0, 1, 2, 3, 5, 7]
    assert [d["kind"] for d in desc] == ["literal"] * 3 + ["match", "match", "match-end"]
    last = desc[-1]
    assert last["match_source_start"] == 0 and last["match_hex"] == b"abra".hex()
    assert last["covers"] == 4 and last["next_hex"] == "" and last["next_char"] is None
    assert desc[3]["next_char"] == "c" and desc[3]["overlapping"] is False
    assert sum(d["covers"] for d in desc) == len(data)
    assert desc[0]["window_start"] == 0 and desc[-1]["lookahead_end"] == len(data)


def test_describe_overlapping_flag_and_limit():
    data = b"aaaa"
    desc = describe_tokens(data, lz77_compress(data))
    assert desc[1]["overlapping"] is True and desc[1]["offset"] == 1 and desc[1]["length"] == 3
    assert len(describe_tokens(b"abc", lz77_compress(b"abc"), limit=2)) == 2


def test_describe_rejects_tokens_not_covering_data():
    with pytest.raises(ValueError):
        describe_tokens(b"abcd", [(0, 0, b"a")])
    with pytest.raises(ValueError):
        describe_tokens(b"a", [(0, 0, b"a"), (0, 0, b"b")])


# ------------------------------------------------------------------ misc properties
def test_compression_is_deterministic():
    data = b"mississippi river " * 30
    assert lz77_compress(data) == lz77_compress(data)


def test_doubling_input_costs_few_extra_tokens_for_repetitive_data():
    one = len(lz77_compress(b"abcdefg" * 100))
    two = len(lz77_compress(b"abcdefg" * 200))
    assert two < 2 * one                                   # repeats are exploited, not re-coded


def test_data_with_no_repeats_never_loses_bytes():
    data = bytes(range(256))
    tokens = lz77_compress(data)
    assert len(tokens) == 256 and all(t[1] == 0 for t in tokens)     # one literal token per byte
    assert compression_stats(256, tokens)["payload_bits"] == 256 * 27


# ------------------------------------------------------------------ matcher contract
@pytest.mark.parametrize("matcher", [_longest_match_reference, _longest_match_fast])
def test_matchers_return_zero_zero_when_nothing_matches(matcher):
    assert matcher(b"ab", 1, 0, 1) == (0, 0)          # 'b' does not occur in the window "a"
    assert matcher(b"ab", 0, 0, 2) == (0, 0)          # empty window at the start
    assert matcher(b"aa", 1, 0, 0) == (0, 0)          # max_len 0 -> nothing can match
    assert matcher(b"abcd", 3, 0, 1) == (0, 0)


@pytest.mark.parametrize("matcher", [_longest_match_reference, _longest_match_fast])
def test_matchers_return_length_and_smallest_offset(matcher):
    assert matcher(b"abab", 2, 0, 2) == (2, 2)
    assert matcher(b"aaaa", 1, 0, 3) == (3, 1)        # overlapping: source runs into the lookahead
    assert matcher(b"abxabyab", 6, 0, 2) == (2, 3)    # ties: offset 3 beats offset 6


def test_overlapping_flag_boundary_offset_equals_length():
    # "abab": tokens a, b, (offset 2, length 2, end). Source [0,2) ends exactly where output
    # begins -> NOT overlapping. "aaaa": offset 1 < length 3 -> overlapping.
    data = b"abab"
    tokens = lz77_compress(data)
    assert tokens[-1] == (2, 2, b"")
    assert describe_tokens(data, tokens)[-1]["overlapping"] is False
    assert describe_tokens(b"aaaa", lz77_compress(b"aaaa"))[-1]["overlapping"] is True
    # offset 2, length 3 -> overlapping by one byte
    d2 = b"ababa"
    assert lz77_compress(d2)[-1] == (2, 3, b"")
    assert describe_tokens(d2, lz77_compress(d2))[-1]["overlapping"] is True

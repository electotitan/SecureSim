"""Checkpoint 3 (Stage 1): Huffman coding tests."""
import itertools
import math
import random

import pytest

from backend.pipeline.source_encoding import (
    HuffmanNode, build_frequency_table, build_huffman_tree, calculate_average_code_length,
    calculate_efficiency_percent, calculate_entropy, generate_codes, huffman_decode,
    huffman_encode, tree_to_dict,
)

SAMPLES = [
    "a", "aaaa", "ab", "aab", "abracadabra", "hello world", "ABBCCC",
    "the quick brown fox jumps over the lazy dog", "  spaces  \n\ttabs\n",
    "héllo wörld", "日本語のテキスト", "emoji 😀😀🎉 mix", "a" * 500 + "b" * 3,
    "\x00\x01\x02 control", "mixed ASCII é 日本 😀",
]


def is_prefix_free(codes):
    vals = list(codes.values())
    return not any(a != b and b.startswith(a) for a in vals for b in vals)


# ------------------------------------------------------------------ frequency table
def test_frequency_table_basic():
    assert build_frequency_table("abracadabra") == {"a": 5, "b": 2, "r": 2, "c": 1, "d": 1}


def test_frequency_table_empty():
    assert build_frequency_table("") == {}


def test_frequency_table_counts_code_points_for_unicode():
    freq = build_frequency_table("é😀😀")
    assert freq == {"é": 1, "😀": 2}
    # a decomposed 'e' + combining acute is two code points, i.e. two symbols
    assert build_frequency_table("e\u0301") == {"e": 1, "\u0301": 1}


def test_frequency_table_rejects_non_str():
    with pytest.raises(TypeError):
        build_frequency_table(b"abc")


# ------------------------------------------------------------------ entropy
def test_entropy_spec_example_is_1_5():
    assert calculate_entropy({"A": 2, "B": 1, "C": 1}) == pytest.approx(1.5)


def test_entropy_uniform_is_log2_k():
    for k in (2, 4, 8, 16):
        freq = {chr(65 + i): 5 for i in range(k)}
        assert calculate_entropy(freq) == pytest.approx(math.log2(k))


def test_entropy_single_symbol_and_empty_are_zero():
    assert calculate_entropy({"a": 10}) == 0.0
    assert calculate_entropy({}) == 0.0
    assert math.copysign(1.0, calculate_entropy({"a": 10})) == 1.0     # not -0.0


def test_entropy_known_value_two_symbols():
    # p = (0.25, 0.75): H = 0.811278...
    assert calculate_entropy({"a": 1, "b": 3}) == pytest.approx(0.8112781244591328)


def test_entropy_bounded_by_log2_k():
    rng = random.Random(7)
    for _ in range(200):
        k = rng.randint(1, 20)
        freq = {chr(65 + i): rng.randint(1, 50) for i in range(k)}
        assert 0.0 <= calculate_entropy(freq) <= math.log2(k) + 1e-12


# ------------------------------------------------------------------ tree and codes
def test_tree_structure_invariants():
    freq = build_frequency_table("the quick brown fox jumps over the lazy dog")
    root = build_huffman_tree(freq)
    assert root.freq == sum(freq.values())
    leaves, stack = [], [root]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            leaves.append(n)
        else:
            assert n.left is not None and n.right is not None          # full binary tree
            assert n.freq == n.left.freq + n.right.freq
            stack += [n.left, n.right]
    assert sorted(l.symbol for l in leaves) == sorted(freq)
    assert len(leaves) == len(freq)


def test_codes_are_prefix_free_and_cover_alphabet():
    for text in SAMPLES:
        _, codes = huffman_encode(text)
        assert set(codes) == set(text)
        assert is_prefix_free(codes)
        assert all(set(c) <= {"0", "1"} and c for c in codes.values())


def test_kraft_equality_for_two_or_more_symbols():
    # Huffman codes are complete: sum 2^-l == 1 exactly.
    for text in SAMPLES:
        _, codes = huffman_encode(text)
        if len(codes) >= 2:
            assert sum(2.0 ** -len(c) for c in codes.values()) == pytest.approx(1.0)


def test_empty_input():
    bits, codes = huffman_encode("")
    assert (bits, codes) == ("", {})
    assert huffman_decode("", {}) == ""
    with pytest.raises(ValueError):
        build_huffman_tree({})


def test_single_symbol_convention():
    bits, codes = huffman_encode("aaaa")
    assert codes == {"a": "0"}
    assert bits == "0000"
    assert huffman_decode(bits, codes) == "aaaa"
    root = build_huffman_tree({"a": 4})
    assert root.is_leaf and generate_codes(root) == {"a": "0"}


def test_deterministic_known_codes():
    # Documented tie-breaking: A(1), B(2), C(3) -> merge A,B first; then C vs (A,B)=3, C wins left.
    _, codes = huffman_encode("ABBCCC")
    assert codes == {"C": "0", "A": "10", "B": "11"}


def test_codes_independent_of_symbol_order_in_text():
    assert huffman_encode("aabbbc")[1] == huffman_encode("cbbbaa")[1] == huffman_encode("bcabab")[1]


def test_repeated_runs_identical():
    for text in SAMPLES:
        assert huffman_encode(text) == huffman_encode(text)


def test_frequent_symbols_get_codes_no_longer_than_rarer_ones():
    freq = build_frequency_table("a" * 50 + "b" * 20 + "c" * 10 + "d" * 5 + "e" * 1)
    codes = generate_codes(build_huffman_tree(freq))
    for x, y in itertools.permutations(freq, 2):
        if freq[x] > freq[y]:
            assert len(codes[x]) <= len(codes[y])


# ------------------------------------------------------------------ optimality
def test_clrs_example_total_cost_224():
    # Classic example (CLRS Fig. 16.3): optimal total = 224 bits per 100 symbols.
    freq = {"a": 45, "b": 13, "c": 12, "d": 16, "e": 9, "f": 5}
    codes = generate_codes(build_huffman_tree(freq))
    assert sum(freq[s] * len(codes[s]) for s in freq) == 224


def _optimal_cost_bruteforce(weights):
    """Minimum sum(w_i * l_i) over ALL complete prefix codes (Kraft sum == 1), by exhaustion."""
    k = len(weights)
    best = None
    for lengths in itertools.product(range(1, k), repeat=k):
        if sum(2.0 ** -l for l in lengths) == 1.0:      # exact: dyadic floats
            cost = sum(w * l for w, l in zip(weights, lengths))
            best = cost if best is None else min(best, cost)
    return best


def test_huffman_cost_equals_bruteforce_optimum_small_alphabets():
    rng = random.Random(99)
    for _ in range(150):
        k = rng.randint(2, 6)
        weights = [rng.randint(1, 30) for _ in range(k)]
        freq = {chr(65 + i): w for i, w in enumerate(weights)}
        codes = generate_codes(build_huffman_tree(freq))
        huffman_cost = sum(freq[s] * len(codes[s]) for s in freq)
        assert huffman_cost == _optimal_cost_bruteforce(weights), weights


def test_source_coding_bounds_H_le_L_lt_H_plus_1():
    for text in SAMPLES:
        freq = build_frequency_table(text)
        if len(freq) < 2:
            continue
        codes = generate_codes(build_huffman_tree(freq))
        h = calculate_entropy(freq)
        l = calculate_average_code_length(freq, codes)
        assert h <= l + 1e-12
        assert l < h + 1


def test_dyadic_distribution_is_100_percent_efficient():
    freq = {"a": 4, "b": 2, "c": 1, "d": 1}                    # p = 1/2, 1/4, 1/8, 1/8
    codes = generate_codes(build_huffman_tree(freq))
    h, l = calculate_entropy(freq), calculate_average_code_length(freq, codes)
    assert h == pytest.approx(1.75) and l == pytest.approx(1.75)
    assert calculate_efficiency_percent(h, l) == pytest.approx(100.0)


def test_efficiency_never_exceeds_100_and_none_when_undefined():
    for text in SAMPLES:
        freq = build_frequency_table(text)
        l = calculate_average_code_length(freq, generate_codes(build_huffman_tree(freq)))
        eff = calculate_efficiency_percent(calculate_entropy(freq), l)
        assert 0.0 <= eff <= 100.0 + 1e-9
    assert calculate_efficiency_percent(0.0, 0.0) is None
    assert calculate_efficiency_percent(0.0, 1.0) == 0.0         # single-symbol convention


def test_average_code_length_empty_and_missing_symbol():
    assert calculate_average_code_length({}, {}) == 0.0
    with pytest.raises(ValueError):
        calculate_average_code_length({"a": 1, "b": 1}, {"a": "0"})


# ------------------------------------------------------------------ round trips
def test_round_trip_samples():
    for text in SAMPLES:
        bits, codes = huffman_encode(text)
        assert huffman_decode(bits, codes) == text


def test_round_trip_random_unicode():
    rng = random.Random(5)
    alphabet = [chr(c) for c in list(range(32, 127)) + [0xE9, 0x3042, 0x4E2D, 0x1F600, 0x1F389]]
    for _ in range(100):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 300)))
        bits, codes = huffman_encode(text)
        assert huffman_decode(bits, codes) == text


def test_encoded_length_equals_sum_of_code_lengths():
    for text in SAMPLES:
        bits, codes = huffman_encode(text)
        assert len(bits) == sum(len(codes[c]) for c in text)


def test_compression_on_skewed_text():
    text = "a" * 900 + "b" * 50 + "c" * 50
    bits, _ = huffman_encode(text)
    assert len(bits) < 8 * len(text.encode("utf-8"))


# ------------------------------------------------------------------ decode validation
def test_decode_rejects_non_binary_bits():
    with pytest.raises(ValueError):
        huffman_decode("01x1", {"a": "0", "b": "1"})
    with pytest.raises(ValueError):
        huffman_decode("0 1", {"a": "0", "b": "1"})


def test_decode_rejects_non_str_bits():
    with pytest.raises(TypeError):
        huffman_decode(b"01", {"a": "0", "b": "1"})


def test_decode_rejects_trailing_incomplete_codeword():
    with pytest.raises(ValueError, match="incomplete"):
        huffman_decode("01", {"a": "0", "b": "10", "c": "11"})


def test_decode_rejects_bits_matching_no_codeword():
    with pytest.raises(ValueError, match="does not match"):
        huffman_decode("1", {"a": "0"})              # incomplete code: '1' leads nowhere


def test_decode_rejects_non_prefix_free_table():
    with pytest.raises(ValueError, match="prefix"):
        huffman_decode("0101", {"a": "0", "b": "01"})


def test_decode_rejects_duplicate_codes():
    with pytest.raises(ValueError, match="duplicate"):
        huffman_decode("0", {"a": "0", "b": "0"})


@pytest.mark.parametrize("table", [
    {"a": ""}, {"a": "02"}, {"a": 0}, {"ab": "0"}, {"": "0"}, {"a": "0", "b": None},
])
def test_decode_rejects_malformed_tables(table):
    with pytest.raises(ValueError):
        huffman_decode("0", table)


def test_decode_nonempty_bits_with_empty_table():
    with pytest.raises(ValueError):
        huffman_decode("0", {})


# ------------------------------------------------------------------ tree validation / export
@pytest.mark.parametrize("freq", [{"a": 0}, {"a": -1}, {"a": 1.5}, {"a": True}, {"ab": 1}, {1: 1}])
def test_build_tree_rejects_bad_frequency_tables(freq):
    with pytest.raises(ValueError):
        build_huffman_tree(freq)


def test_generate_codes_rejects_non_node():
    with pytest.raises(TypeError):
        generate_codes({"a": 1})


def test_tree_to_dict_is_json_compatible_and_consistent():
    import json
    freq = build_frequency_table("abracadabra")
    root = build_huffman_tree(freq)
    d = tree_to_dict(root)
    json.dumps(d)
    assert d["freq"] == sum(freq.values())
    leaves, stack = {}, [d]
    while stack:
        n = stack.pop()
        if n["left"] is None and n["right"] is None:
            leaves[n["symbol"]] = n["freq"]
        else:
            assert n["freq"] == n["left"]["freq"] + n["right"]["freq"]
            stack += [n["left"], n["right"]]
    assert leaves == freq


def test_tree_to_dict_single_leaf():
    assert tree_to_dict(build_huffman_tree({"z": 3})) == {"symbol": "z", "freq": 3, "left": None, "right": None}


# ------------------------------------------------------------------ tie-breaking contract
def test_tiebreak_equal_frequency_leaves_ordered_by_code_point():
    # a,b,c,d all freq 1. Leaves numbered by code point: a0 b1 c2 d3.
    # merge(a,b)=N4(2); merge(c,d)=N5(2); merge(N4,N5): N4 popped first -> left.
    _, codes = huffman_encode("dcba")
    assert codes == {"a": "00", "b": "01", "c": "10", "d": "11"}


def test_tiebreak_leaf_beats_internal_node_of_equal_weight():
    # a:1 b:1 c:2 -> N3(2)=merge(a,b); then c(2, order 2) ties with N3(2, order 3): leaf c first.
    _, codes = huffman_encode("abcc")
    assert codes == {"c": "0", "a": "10", "b": "11"}


def test_tiebreak_independent_of_dict_insertion_order():
    freq1 = {"x": 2, "a": 2, "m": 1, "z": 1}
    freq2 = dict(reversed(list(freq1.items())))
    assert generate_codes(build_huffman_tree(freq1)) == generate_codes(build_huffman_tree(freq2))

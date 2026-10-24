"""Checkpoint 8: CRC-32 tests.

zlib.crc32 is used HERE ONLY as an independent reference implementation; a static test verifies that the
implementation module itself imports no zlib / binascii / crcmod / hashlib.
"""
import ast
import itertools
import pathlib
import random
import zlib

import pytest

from backend.pipeline import error_detection_crc as crc
from backend.pipeline.error_detection_crc import (
    CHECK_VALUE, POLY_FULL, POLY_NORMAL, POLY_REFLECTED, RESIDUE, crc32_bitwise, crc32_compute,
    crc32_forge_suffix, crc32_polynomial_division, crc32_state, crc32_table, crc32_trace, crc32_verify,
    error_span, flip_bits, generate_table, polynomial_expression, polynomial_terms,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def rand_bytes(rng, n):
    return bytes(rng.randrange(256) for _ in range(n))


# ================================================================== CRITICAL: the standard check value
def test_standard_check_value_123456789():
    assert crc32_compute(b"123456789") == 0xCBF43926
    assert crc32_compute("123456789".encode("ascii")) == CHECK_VALUE == 0xCBF43926


@pytest.mark.parametrize("data, expected", [
    (b"", 0x00000000),
    (b"a", 0xE8B7BE43),
    (b"abc", 0x352441C2),
    (b"message digest", 0x20159D7F),
    (b"abcdefghijklmnopqrstuvwxyz", 0x4C2750BD),
    (b"The quick brown fox jumps over the lazy dog", 0x414FA339),
    (b"hello world", 0x0D4A1185),
    (b"\x00", 0xD202EF8D),
    (b"\xff\xff\xff\xff", 0xFFFFFFFF),
    (bytes(32), 0x190A55AD),
])
def test_published_vectors(data, expected):
    assert crc32_compute(data) == expected


def test_matches_independent_reference_implementation_on_random_data():
    rng = random.Random(1)
    for n in list(range(0, 100)) + [255, 256, 1000, 4096, 65537]:
        d = rand_bytes(rng, n)
        assert crc32_compute(d) == zlib.crc32(d)


# ================================================================== table derivation
def test_polynomial_constants_and_derivation():
    assert POLY_NORMAL == 0x04C11DB7 and POLY_FULL == 0x104C11DB7 and POLY_REFLECTED == 0xEDB88320
    assert crc._reflect32(POLY_NORMAL) == POLY_REFLECTED                  # the reflected form is the bit-reversal
    assert crc._reflect32(POLY_REFLECTED) == POLY_NORMAL                  # and reflection is an involution
    assert crc._reflect32(0x00000001) == 0x80000000 and crc._reflect32(0x12345678) == 0x1E6A2C48


def test_table_is_derived_and_has_the_known_entries():
    t = crc32_table()
    assert isinstance(t, tuple) and len(t) == 256 and all(0 <= x <= 0xFFFFFFFF for x in t)
    assert (t[0], t[1], t[2], t[3], t[255]) == (0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x2D02EF8D)
    assert t == generate_table() == generate_table(0xEDB88320)


def test_table_entries_follow_the_shift_xor_definition_independently():
    for i in range(256):
        reg = i
        for _ in range(8):                                     # written out again, not by calling the module
            lsb = reg & 1
            reg >>= 1
            if lsb:
                reg ^= 0xEDB88320
        assert crc32_table()[i] == reg


def test_table_is_linear_over_gf2():
    t = crc32_table()
    for a, b in itertools.product(range(0, 256, 7), repeat=2):
        assert t[a ^ b] == t[a] ^ t[b]


def test_top_bytes_of_the_table_are_a_permutation():
    assert sorted(e >> 24 for e in crc32_table()) == list(range(256))


def test_generated_table_works_for_a_different_polynomial_crc32c_castagnoli():
    table = generate_table(0x82F63B78)                          # reflected form of the Castagnoli polynomial 0x1EDC6F41
    assert table[1] == 0xF26B8303
    reg = 0xFFFFFFFF
    for byte in b"123456789":
        reg = table[(reg ^ byte) & 0xFF] ^ (reg >> 8)
    assert reg ^ 0xFFFFFFFF == 0xE3069283                       # published CRC-32C check value


def test_generate_table_validation():
    for bad in (-1, 2 ** 32, 1.5, "x", True, None):
        with pytest.raises((TypeError, ValueError)):
            generate_table(bad)


# ================================================================== three independent formulations
def test_three_formulations_agree_for_every_short_length():
    rng = random.Random(2)
    for n in range(0, 71):
        d = rand_bytes(rng, n)
        v = crc32_compute(d)
        assert v == crc32_bitwise(d) == crc32_polynomial_division(d), n


def test_three_formulations_agree_on_larger_random_inputs():
    rng = random.Random(3)
    for _ in range(40):
        d = rand_bytes(rng, rng.randint(100, 3000))
        assert crc32_compute(d) == crc32_bitwise(d) == crc32_polynomial_division(d)


def test_every_single_byte_input_agrees_across_formulations_and_the_reference():
    for b in range(256):
        d = bytes([b])
        assert crc32_compute(d) == crc32_bitwise(d) == crc32_polynomial_division(d) == zlib.crc32(d)


def test_polynomial_division_view_matches_for_the_check_string():
    assert crc32_polynomial_division(b"123456789") == 0xCBF43926 == crc32_bitwise(b"123456789")


def test_leading_zero_bytes_change_the_crc_because_init_is_all_ones():
    assert len({crc32_compute(bytes(k)) for k in range(1, 20)}) == 19
    assert crc32_compute(b"\x00" + b"abc") != crc32_compute(b"abc")


# ================================================================== incremental use and verification
def test_incremental_computation_matches_one_shot():
    rng = random.Random(4)
    for _ in range(200):
        d = rand_bytes(rng, rng.randint(0, 300))
        cut = rng.randint(0, len(d))
        assert crc32_compute(d[cut:], crc32_compute(d[:cut])) == crc32_compute(d)
    d = rand_bytes(rng, 100)
    running = 0
    for i in range(0, 100, 7):
        running = crc32_compute(d[i:i + 7], running)
    assert running == crc32_compute(d)


def test_state_function_relation():
    for d in (b"", b"abc", bytes(range(50))):
        assert crc32_compute(d) == crc32_state(d) ^ 0xFFFFFFFF
        assert crc32_state(d, 0xFFFFFFFF) == crc32_state(d)
    assert crc32_state(b"") == 0xFFFFFFFF and crc32_compute(b"") == 0


def test_verify_accepts_correct_and_rejects_wrong_checksums():
    d = b"The quick brown fox"
    good = crc32_compute(d)
    assert crc32_verify(d, good) is True
    assert crc32_verify(d, good ^ 1) is False and crc32_verify(d, 0) is False and crc32_verify(d + b"!", good) is False
    assert crc32_verify(b"123456789", 0xCBF43926) is True and crc32_verify(b"", 0) is True


def test_verify_and_compute_validation():
    for bad in (-1, 2 ** 32, 2 ** 40):
        with pytest.raises(ValueError):
            crc32_verify(b"abc", bad)
        with pytest.raises(ValueError):
            crc32_compute(b"abc", bad)
    for bad in (1.5, "1", None, True, [1]):
        with pytest.raises(TypeError):
            crc32_verify(b"abc", bad)
        with pytest.raises(TypeError):
            crc32_compute(b"abc", bad)
    functions = [crc32_compute, crc32_bitwise, crc32_polynomial_division, crc32_state, crc32_trace,
                 lambda d: flip_bits(d, []), lambda d: crc32_forge_suffix(d, 0)]
    for bad in ("abc", 5, None, [1, 2]):
        for fn in functions:
            with pytest.raises(TypeError):
                fn(bad)
        with pytest.raises(TypeError):
            crc32_verify(bad, 0)


def test_bytearray_inputs_are_accepted():
    assert crc32_compute(bytearray(b"123456789")) == 0xCBF43926 == crc32_bitwise(bytearray(b"123456789"))


# ================================================================== polynomial representation
def test_crc32_polynomial_terms_and_expression():
    assert polynomial_terms() == [32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 4, 2, 1, 0]
    assert polynomial_expression() == "x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1"
    assert len(polynomial_terms()) == 15                       # odd term count -> not divisible by (x + 1)


def test_polynomial_terms_for_other_polynomials():
    assert polynomial_expression(0x11021) == "x^16 + x^12 + x^5 + 1"          # CRC-16-CCITT
    assert polynomial_terms(0b1011) == [3, 1, 0] and polynomial_expression(1) == "1" and polynomial_expression(2) == "x"
    for bad in (0, -5, 1.5, "x", True):
        with pytest.raises(ValueError):
            polynomial_terms(bad)


def test_generator_polynomial_is_not_divisible_by_x_plus_1_so_no_odd_error_guarantee():
    assert bin(POLY_FULL).count("1") % 2 == 1                  # G(1) = 1 mod 2


# ================================================================== trace
def test_trace_structure_and_register_chain():
    d = b"123456789"
    tr = crc32_trace(d)
    assert tr["initial_register"] == 0xFFFFFFFF and tr["final_xor"] == 0xFFFFFFFF
    assert tr["checksum"] == 0xCBF43926 == tr["final_register"] ^ 0xFFFFFFFF
    steps = tr["byte_steps"]
    assert len(steps) == 9 and tr["steps_total"] == 9 and tr["steps_returned"] == 9 and tr["steps_truncated"] is False
    assert steps[0]["register_before"] == 0xFFFFFFFF
    for a, b in zip(steps, steps[1:]):
        assert b["register_before"] == a["register_after"]
    assert steps[-1]["register_after"] == tr["final_register"]
    for i, s in enumerate(steps):
        assert s["index"] == i and s["byte"] == d[i]
        assert s["table_index"] == (s["register_before"] ^ s["byte"]) & 0xFF
        assert s["table_value"] == crc32_table()[s["table_index"]]
        assert s["register_after"] == s["table_value"] ^ (s["register_before"] >> 8)


def test_trace_polynomial_fields():
    tr = crc32_trace(b"x")
    assert (tr["polynomial_normal"], tr["polynomial_reflected"], tr["polynomial_full"]) == (0x04C11DB7, 0xEDB88320, 0x104C11DB7)
    assert tr["polynomial_terms"] == polynomial_terms() and tr["polynomial_expression"].startswith("x^32 + x^26")


def test_trace_bit_steps_reproduce_the_table_step():
    tr = crc32_trace(b"123456789", bit_steps_bytes=3)
    assert len(tr["bit_steps"]) == 3
    for bs, st in zip(tr["bit_steps"], tr["byte_steps"]):
        steps = bs["steps"]
        assert len(steps) == 9 and steps[0]["step"] == 0 and steps[0]["lsb"] is None
        assert steps[0]["register"] == st["register_before"] ^ st["byte"]
        reg = steps[0]["register"]
        for s in steps[1:]:
            lsb = reg & 1
            reg = (reg >> 1) ^ 0xEDB88320 if lsb else reg >> 1
            assert s["lsb"] == lsb and s["xor_polynomial"] is bool(lsb) and s["register"] == reg
        assert steps[-1]["register"] == st["register_after"]
        assert bs["byte_index"] == st["index"] and bs["byte"] == st["byte"]


def test_trace_bounds_and_truncation():
    d = bytes(range(100))
    tr = crc32_trace(d, max_byte_steps=10, bit_steps_bytes=0)
    assert len(tr["byte_steps"]) == 10 and tr["steps_total"] == 100 and tr["steps_truncated"] is True
    assert tr["bit_steps"] == [] and tr["checksum"] == crc32_compute(d)               # truncation never changes the result
    assert crc32_trace(d, max_byte_steps=0)["byte_steps"] == [] and crc32_trace(d, max_byte_steps=100)["steps_truncated"] is False
    assert len(crc32_trace(d, bit_steps_bytes=500)["bit_steps"]) == 100


def test_trace_of_empty_data():
    tr = crc32_trace(b"")
    assert tr["byte_steps"] == [] and tr["bit_steps"] == [] and tr["final_register"] == 0xFFFFFFFF and tr["checksum"] == 0
    assert tr["steps_total"] == 0 and tr["steps_truncated"] is False


@pytest.mark.parametrize("bad", [-1, 1.5, "3", None, True])
def test_trace_parameter_validation(bad):
    with pytest.raises(ValueError):
        crc32_trace(b"abc", max_byte_steps=bad)
    with pytest.raises(ValueError):
        crc32_trace(b"abc", bit_steps_bytes=bad)


def test_trace_is_json_compatible():
    import json
    json.dumps(crc32_trace(b"123456789", bit_steps_bytes=9))


# ================================================================== error detection guarantees
def test_every_single_bit_error_is_detected():
    rng = random.Random(5)
    for n in (1, 2, 8, 64):
        d = rand_bytes(rng, n)
        base = crc32_compute(d)
        for p in range(8 * n):
            assert crc32_compute(flip_bits(d, [p])) != base


def test_every_two_bit_error_in_a_32_byte_message_is_detected():
    d = bytes(range(32))
    base = crc32_compute(d)
    for p, q in itertools.combinations(range(256), 2):
        assert crc32_compute(flip_bits(d, [p, q])) != base


def test_every_three_bit_error_in_an_8_byte_message_is_detected():
    d = b"ABCDEFGH"
    base = crc32_compute(d)
    for combo in itertools.combinations(range(64), 3):
        assert crc32_compute(flip_bits(d, combo)) != base


def test_every_burst_up_to_32_bits_is_detected():
    rng = random.Random(6)
    for _ in range(3000):
        d = rand_bytes(rng, rng.randint(8, 40))
        length = rng.randint(1, 32)
        start = rng.randint(0, 8 * len(d) - length)
        positions = {start, start + length - 1} | {p for p in range(start, start + length) if rng.random() < 0.5}
        assert error_span(sorted(positions)) == length
        assert crc32_compute(flip_bits(d, sorted(positions))) != crc32_compute(d)


def test_solid_bursts_of_every_length_1_to_32_at_every_position_are_detected():
    d = bytes(range(16))
    base = crc32_compute(d)
    for length in range(1, 33):
        for start in range(0, 8 * len(d) - length + 1):
            assert crc32_compute(flip_bits(d, range(start, start + length))) != base


def test_a_33_bit_burst_shaped_like_the_generator_polynomial_is_NOT_detected():
    # An error polynomial that is a multiple of G(x) is invisible. Message bit p (transmission order) is the
    # coefficient of x^(n-1-p) of M(x), so the error G(x)*x^k flips bits p = n-1-(e+k) for each exponent e of G.
    d = bytes(range(8))
    n, k = 8 * len(d), 5
    positions = sorted(n - 1 - (e + k) for e in polynomial_terms())
    assert error_span(positions) == 33 and len(positions) == 15
    corrupted = flip_bits(d, positions)
    assert corrupted != d and crc32_compute(corrupted) == crc32_compute(d)             # corruption slips through
    for kk in range(0, n - 32):                                                        # at every shift, in every message
        pos = sorted(n - 1 - (e + kk) for e in polynomial_terms())
        assert crc32_compute(flip_bits(d, pos)) == crc32_compute(d)


def test_undetected_errors_exist_only_for_multiples_of_the_generator():
    # Random 33+ bit patterns almost never collide: no escape among 20000 random multi-bit errors.
    rng = random.Random(7)
    d = rand_bytes(rng, 64)
    base = crc32_compute(d)
    for _ in range(20000):
        pos = rng.sample(range(512), rng.randint(2, 40))
        assert crc32_compute(flip_bits(d, pos)) != base


def test_crc_is_affine_over_gf2():
    rng = random.Random(8)
    for _ in range(300):
        n = rng.randint(1, 80)
        a, b, c = rand_bytes(rng, n), rand_bytes(rng, n), rand_bytes(rng, n)
        x = bytes(i ^ j ^ k for i, j, k in zip(a, b, c))
        assert crc32_compute(x) == crc32_compute(a) ^ crc32_compute(b) ^ crc32_compute(c)


def test_residue_constant_after_appending_the_crc_little_endian():
    rng = random.Random(9)
    assert RESIDUE == 0x2144DF1C
    for _ in range(200):
        d = rand_bytes(rng, rng.randint(0, 100))
        assert crc32_compute(d + crc32_compute(d).to_bytes(4, "little")) == RESIDUE
    assert crc32_compute(b"123456789" + (0xCBF43926).to_bytes(4, "little")) == RESIDUE


def test_crc_detects_reordering_that_the_16_bit_internet_checksum_cannot():
    from backend.pipeline.packetization import internet_checksum
    a = bytes.fromhex("12345678 9abcdef0".replace(" ", ""))
    b = bytes.fromhex("9abc1234 def05678".replace(" ", ""))                       # same 16-bit words, different order
    assert internet_checksum(a) == internet_checksum(b)
    assert crc32_compute(a) != crc32_compute(b)
    # ... and payload tampering (not covered by the packet header checksum) is detected too
    assert crc32_compute(b"ORIGINAL") != crc32_compute(b"TAMPERED")


# ================================================================== forging: CRC is not authentication
def test_forging_hits_any_target_for_random_data():
    rng = random.Random(10)
    for _ in range(500):
        d = rand_bytes(rng, rng.randint(0, 60))
        target = rng.randrange(2 ** 32)
        suffix = crc32_forge_suffix(d, target)
        assert isinstance(suffix, bytes) and len(suffix) == 4
        assert crc32_compute(d + suffix) == target == zlib.crc32(d + suffix)


def test_forging_extreme_targets_and_empty_data():
    for d in (b"", b"a", bytes(range(200))):
        for target in (0, 1, 0xFFFFFFFF, 0x80000000, 0x00000100, 0xDEADBEEF):
            assert crc32_compute(d + crc32_forge_suffix(d, target)) == target


def test_forged_message_collides_with_the_original_checksum():
    original = b"Transfer 100 EUR to account 1234"
    tampered = b"Transfer 999 EUR to account 6666"
    assert crc32_compute(original) != crc32_compute(tampered)
    forged = tampered + crc32_forge_suffix(tampered, crc32_compute(original))
    assert forged != original and crc32_verify(forged, crc32_compute(original)) is True      # the attacker repaired the CRC


def test_forging_is_deterministic_and_keeps_the_data_untouched():
    d = b"payload"
    assert crc32_forge_suffix(d, 123) == crc32_forge_suffix(d, 123)
    assert crc32_forge_suffix(d, 123) != crc32_forge_suffix(d, 124)
    assert d == b"payload"


@pytest.mark.parametrize("bad", [-1, 2 ** 32])
def test_forging_target_range(bad):
    with pytest.raises(ValueError):
        crc32_forge_suffix(b"x", bad)


@pytest.mark.parametrize("bad", [1.5, "1", None, True])
def test_forging_target_type(bad):
    with pytest.raises(TypeError):
        crc32_forge_suffix(b"x", bad)


def test_forging_data_type():
    for bad in ("abc", 5, None):
        with pytest.raises(TypeError):
            crc32_forge_suffix(bad, 0)


# ================================================================== flip_bits / error_span helpers
def test_flip_bits_bit_order_is_lsb_first_within_each_byte():
    assert flip_bits(b"\x00", [0]) == b"\x01" and flip_bits(b"\x00", [7]) == b"\x80"
    assert flip_bits(b"\x00\x00", [8]) == b"\x00\x01" and flip_bits(b"\x00\x00", [15]) == b"\x00\x80"
    assert flip_bits(b"\xff", [0, 1]) == b"\xfc"
    assert flip_bits(b"abc", []) == b"abc"


def test_flip_bits_is_an_xor_mask_so_repeats_cancel_and_it_is_an_involution():
    assert flip_bits(b"abc", [3, 3]) == b"abc"
    assert flip_bits(b"abc", [3, 3, 3]) == flip_bits(b"abc", [3])
    d = bytes(range(10))
    pos = [1, 9, 33, 70]
    assert flip_bits(flip_bits(d, pos), pos) == d


def test_flip_bits_validation_and_no_mutation():
    d = b"abc"
    for bad in (-1, 24, 100):
        with pytest.raises(ValueError, match="outside the data"):
            flip_bits(d, [bad])
    for bad in (1.5, "1", None, True):
        with pytest.raises(TypeError):
            flip_bits(d, [bad])
    with pytest.raises(ValueError):
        flip_bits(b"", [0])
    ba = bytearray(b"abc")
    flip_bits(ba, [0])
    assert ba == b"abc"


def test_error_span():
    assert error_span([]) is None and error_span([5]) == 1 and error_span([3, 10]) == 8 and error_span([10, 3, 7]) == 8


# ================================================================== independence of the implementation
def test_implementation_uses_no_reference_or_third_party_crc_libraries():
    tree = ast.parse((ROOT / "backend" / "pipeline" / "error_detection_crc.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "typing"}, imported
    for banned in ("zlib", "binascii", "crcmod", "hashlib", "struct"):
        assert banned not in imported


def test_the_module_stores_no_literal_crc_table():
    src = (ROOT / "backend" / "pipeline" / "error_detection_crc.py").read_text().lower()
    for entry in ("0x77073096", "0xee0e612c", "0x990951ba", "0x2d02ef8d"):
        assert entry not in src

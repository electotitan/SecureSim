"""Checkpoint 2: GF(2^8) arithmetic and S-box derivation tests.

The standard S-box below is an independent REFERENCE (FIPS-197, Figure 7) used only to
check the generated table. The implementation never reads it.
"""
import inspect
import random

import pytest

from backend.pipeline import gf256
from backend.pipeline.gf256 import (
    AES_POLY, affine_transform, generate_sbox, gf_add, gf_inverse,
    gf_inverse_by_exponentiation, gf_inverse_with_trace, gf_multiply, gf_pow,
    poly_degree, poly_divmod_gf2, poly_multiply_gf2, sbox_table,
)

_REF_ROWS = """
63 7c 77 7b f2 6b 6f c5 30 01 67 2b fe d7 ab 76
ca 82 c9 7d fa 59 47 f0 ad d4 a2 af 9c a4 72 c0
b7 fd 93 26 36 3f f7 cc 34 a5 e5 f1 71 d8 31 15
04 c7 23 c3 18 96 05 9a 07 12 80 e2 eb 27 b2 75
09 83 2c 1a 1b 6e 5a a0 52 3b d6 b3 29 e3 2f 84
53 d1 00 ed 20 fc b1 5b 6a cb be 39 4a 4c 58 cf
d0 ef aa fb 43 4d 33 85 45 f9 02 7f 50 3c 9f a8
51 a3 40 8f 92 9d 38 f5 bc b6 da 21 10 ff f3 d2
cd 0c 13 ec 5f 97 44 17 c4 a7 7e 3d 64 5d 19 73
60 81 4f dc 22 2a 90 88 46 ee b8 14 de 5e 0b db
e0 32 3a 0a 49 06 24 5c c2 d3 ac 62 91 95 e4 79
e7 c8 37 6d 8d d5 4e a9 6c 56 f4 ea 65 7a ae 08
ba 78 25 2e 1c a6 b4 c6 e8 dd 74 1f 4b bd 8b 8a
70 3e b5 66 48 03 f6 0e 61 35 57 b9 86 c1 1d 9e
e1 f8 98 11 69 d9 8e 94 9b 1e 87 e9 ce 55 28 df
8c a1 89 0d bf e6 42 68 41 99 2d 0f b0 54 bb 16
"""
REFERENCE_SBOX = [int(tok, 16) for tok in _REF_ROWS.split()]


# ------------------------------------------------------------------ reference sanity
def test_reference_table_is_well_formed():
    # Guards against a typo in the reference itself (it must be a permutation of 0..255).
    assert len(REFERENCE_SBOX) == 256
    assert sorted(REFERENCE_SBOX) == list(range(256))


# ------------------------------------------------------------------ GF(2)[x] helpers
def test_poly_degree():
    assert poly_degree(0) == -1
    assert poly_degree(1) == 0
    assert poly_degree(0x11B) == 8


def test_poly_multiply_known():
    assert poly_multiply_gf2(0b11, 0b11) == 0b101      # (x+1)^2 = x^2 + 1 over GF(2)
    assert poly_multiply_gf2(0, 0xFF) == 0
    assert poly_multiply_gf2(0x57, 1) == 0x57


def test_poly_divmod_reconstructs_dividend():
    rng = random.Random(1)
    for _ in range(500):
        a, b = rng.randrange(0, 1 << 16), rng.randrange(1, 1 << 9)
        q, r = poly_divmod_gf2(a, b)
        assert poly_multiply_gf2(q, b) ^ r == a
        assert poly_degree(r) < poly_degree(b)


def test_poly_divmod_by_zero_rejected():
    with pytest.raises(ZeroDivisionError):
        poly_divmod_gf2(5, 0)


def test_aes_polynomial_is_irreducible():
    # No polynomial of degree 1..4 divides m(x) -> irreducible (a degree-8 reducible
    # polynomial must have a factor of degree <= 4).
    for d in range(1, 5):
        for p in range(1 << d, 1 << (d + 1)):
            assert poly_divmod_gf2(AES_POLY, p)[1] != 0, f"0x{p:x} divides m(x)"


# ------------------------------------------------------------------ multiplication
def test_multiply_fips197_examples():
    assert gf_multiply(0x57, 0x83) == 0xC1      # FIPS-197 section 4.2
    assert gf_multiply(0x57, 0x13) == 0xFE      # FIPS-197 section 4.2


def test_multiply_xtime_chain_fips197():
    a = 0x57
    assert gf_multiply(a, 0x02) == 0xAE
    assert gf_multiply(a, 0x04) == 0x47
    assert gf_multiply(a, 0x08) == 0x8E
    assert gf_multiply(a, 0x10) == 0x07


def test_multiply_zero_and_identity():
    for a in range(256):
        assert gf_multiply(a, 0) == 0
        assert gf_multiply(0, a) == 0
        assert gf_multiply(a, 1) == a
        assert gf_multiply(1, a) == a


def test_multiply_commutative_exhaustive():
    for a in range(256):
        for b in range(a, 256):
            assert gf_multiply(a, b) == gf_multiply(b, a)


def test_multiply_associative_and_distributive_random():
    rng = random.Random(2024)
    for _ in range(20000):
        a, b, c = (rng.randrange(256) for _ in range(3))
        assert gf_multiply(gf_multiply(a, b), c) == gf_multiply(a, gf_multiply(b, c))
        assert gf_multiply(a, gf_add(b, c)) == gf_add(gf_multiply(a, b), gf_multiply(a, c))


def test_multiply_closed_in_range():
    for a in range(0, 256, 7):
        for b in range(256):
            assert 0 <= gf_multiply(a, b) <= 255


def test_no_zero_divisors():
    for a in range(1, 256):
        for b in range(1, 256):
            assert gf_multiply(a, b) != 0


def test_generator_3_has_order_255():
    # 0x03 generates the multiplicative group: its powers hit all 255 nonzero elements.
    seen, x = set(), 1
    for _ in range(255):
        seen.add(x)
        x = gf_multiply(x, 3)
    assert seen == set(range(1, 256))
    assert x == 1


def test_gf_pow_matches_repeated_multiplication():
    for a in (0, 1, 2, 3, 0x53, 0xFF):
        acc = 1
        for n in range(0, 20):
            assert gf_pow(a, n) == acc
            acc = gf_multiply(acc, a)


def test_fermat_a_pow_255_is_one():
    for a in range(1, 256):
        assert gf_pow(a, 255) == 1


# ------------------------------------------------------------------ inversion
def test_inverse_all_255_nonzero_elements():
    for a in range(1, 256):
        inv = gf_inverse(a)
        assert 1 <= inv <= 255
        assert gf_multiply(a, inv) == 1
        assert gf_multiply(inv, a) == 1


def test_inverse_is_involution_and_unique():
    invs = [gf_inverse(a) for a in range(1, 256)]
    assert sorted(invs) == list(range(1, 256))          # a -> a^-1 is a bijection
    for a in range(1, 256):
        assert gf_inverse(gf_inverse(a)) == a


def test_inverse_known_values():
    assert gf_inverse(1) == 1
    assert gf_inverse(0x53) == 0xCA                     # FIPS-197 / Rijndael worked example
    assert gf_inverse(0x02) == 0x8D


def test_inverse_euclid_agrees_with_exponentiation_reference():
    for a in range(1, 256):
        assert gf_inverse(a) == gf_inverse_by_exponentiation(a)


def test_inverse_of_zero_rejected():
    with pytest.raises(ValueError):
        gf_inverse(0)
    with pytest.raises(ValueError):
        gf_inverse_by_exponentiation(0)


def test_inverse_trace_is_consistent():
    inv, steps = gf_inverse_with_trace(0x53)
    assert inv == 0xCA
    assert steps and steps[-1]["remainder"] == 0
    for s in steps:
        # each step is a valid division and the Bezout coefficient update holds
        assert poly_multiply_gf2(s["quotient"], s["divisor"]) ^ s["remainder"] == s["dividend"]
        assert s["t_next"] == s["t_prev"] ^ poly_multiply_gf2(s["quotient"], s["t_curr"])
    assert steps[0]["dividend"] == AES_POLY


# ------------------------------------------------------------------ input validation
@pytest.mark.parametrize("bad", [-1, 256, 1000])
def test_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        gf_multiply(bad, 1)
    with pytest.raises(ValueError):
        gf_multiply(1, bad)
    with pytest.raises(ValueError):
        gf_inverse(bad)


@pytest.mark.parametrize("bad", [1.5, "a", None, True])
def test_non_int_rejected(bad):
    with pytest.raises(TypeError):
        gf_multiply(bad, 1)
    with pytest.raises(TypeError):
        gf_inverse(bad)


# ------------------------------------------------------------------ S-box
def test_affine_known_values():
    assert affine_transform(0x00) == 0x63               # S(0) = Affine(0)
    assert affine_transform(0x01) == 0x7C               # S(1) = Affine(inverse of 1 = 1)


def test_affine_matches_rotation_formulation():
    # Independent formulation: b ^ rotl1 ^ rotl2 ^ rotl3 ^ rotl4 ^ 0x63.
    def rotl(x, n):
        return ((x << n) | (x >> (8 - n))) & 0xFF
    for b in range(256):
        expected = b ^ rotl(b, 1) ^ rotl(b, 2) ^ rotl(b, 3) ^ rotl(b, 4) ^ 0x63
        assert affine_transform(b) == expected


def test_sbox_length_and_range():
    sb = generate_sbox()
    assert len(sb) == 256
    assert all(isinstance(v, int) and 0 <= v <= 255 for v in sb)


def test_sbox_is_bijective():
    assert sorted(generate_sbox()) == list(range(256))


def test_sbox_matches_all_256_standard_entries():
    sb = generate_sbox()
    mismatches = [(i, sb[i], REFERENCE_SBOX[i]) for i in range(256) if sb[i] != REFERENCE_SBOX[i]]
    assert mismatches == []


def test_sbox_spot_values_fips197():
    sb = generate_sbox()
    assert sb[0x00] == 0x63
    assert sb[0x01] == 0x7C
    assert sb[0x53] == 0xED                             # FIPS-197 section 5.1.1 example
    assert sb[0xFF] == 0x16


def test_sbox_has_no_fixed_or_opposite_fixed_points():
    # Design property of the AES S-box: S(a) != a and S(a) != ~a for every a.
    sb = generate_sbox()
    for a in range(256):
        assert sb[a] != a
        assert sb[a] != a ^ 0xFF


def test_sbox_table_cached_matches_generate():
    assert list(sbox_table()) == generate_sbox()
    assert sbox_table() is sbox_table()


def test_sbox_is_not_hardcoded_in_implementation():
    # Cheap guard: the module source must not embed the first standard S-box bytes.
    src = inspect.getsource(gf256).lower().replace(" ", "").replace("\n", "")
    assert "0x63,0x7c,0x77" not in src
    assert "63,7c,77,7b" not in src

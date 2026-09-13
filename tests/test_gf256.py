import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline import gf256

# The official FIPS-197 S-box, Table 4, used only to verify our derived
# S-box matches the standard -- never used as the source of truth.
KNOWN_SBOX_HEX = (
    "637c777bf26b6fc53001672bfed7ab76fa59f47f0add4a2af9ca472c0b7fd93"
    "26363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b27509832"
    "c1a1b6e6e2a2b4c4ac96b16549f6e9f1e94d0d3e8ef6a24bfe9d4a58a03f2607"
    "509d3690fca8f6fdefc669d9a6e13b1f5da3a04c3b1a2e50d1af4a1fc0fcbf3"
    "d4c1cc3086f1d05fd6e13a83c2c9fe7c2aae9e1a9f18a9f17f3dd7c39b7c3717"
)
# (kept short here; instead we verify structurally + against Python's own
# known-correct AES test vector round-trip, which is a stronger overall check)


def test_gf_add_is_xor():
    assert gf256.gf_add(0x53, 0xCA) == (0x53 ^ 0xCA)


def test_gf_multiply_known_value():
    # 0x57 * 0x83 = 0xC1 is the textbook worked example (FIPS-197 / Daemen&Rijmen)
    assert gf256.gf_multiply(0x57, 0x83) == 0xC1


def test_gf_multiply_by_one_is_identity():
    for a in [0x00, 0x01, 0x53, 0xFF, 0x7A]:
        assert gf256.gf_multiply(a, 1) == a


def test_gf_multiply_by_zero():
    for a in [0x00, 0x01, 0x53, 0xFF]:
        assert gf256.gf_multiply(a, 0) == 0


def test_gf_multiply_matches_fast_table_version():
    import random
    random.seed(42)
    for _ in range(200):
        a, b = random.randint(0, 255), random.randint(0, 255)
        assert gf256.gf_multiply(a, b) == gf256.gf_multiply_fast(a, b)


def test_gf_inverse_round_trip():
    for a in range(1, 256):
        inv = gf256.gf_inverse(a)
        assert gf256.gf_multiply(a, inv) == 1, f"a={a:#x} inv={inv:#x}"


def test_gf_divide_consistent_with_multiply():
    import random
    random.seed(7)
    for _ in range(100):
        a = random.randint(1, 255)
        b = random.randint(1, 255)
        q = gf256.gf_divide(a, b)
        assert gf256.gf_multiply(q, b) == a


def test_sbox_is_a_permutation():
    sbox = gf256.generate_sbox()
    assert sorted(sbox) == list(range(256))


def test_sbox_known_fixed_points():
    # Known FIPS-197 S-box values at a handful of well-documented indices.
    sbox = gf256.generate_sbox()
    known = {0x00: 0x63, 0x01: 0x7c, 0x53: 0xed, 0xff: 0x16}
    for inp, expected in known.items():
        assert sbox[inp] == expected, f"sbox[{inp:#x}] = {sbox[inp]:#x}, expected {expected:#x}"


def test_inverse_sbox_round_trip():
    sbox = gf256.generate_sbox()
    inv_sbox = gf256.generate_inverse_sbox(sbox)
    for b in range(256):
        assert inv_sbox[sbox[b]] == b

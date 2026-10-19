"""Checkpoint 5: AES-128 (encryption, decryption, CBC, padding) tests.

Official vectors used (published values, independent of the implementation):
  * FIPS-197 Appendix C.1 (key 000102...0f) and Appendix B / A.1 (key 2b7e1516...)
  * NIST SP 800-38A F.1.1 (ECB-AES128) and F.2.1 (CBC-AES128)
"""
import random

import pytest

from backend.pipeline.decryption_aes import (
    INV_MIX_MATRIX, aes_cbc_decrypt, aes_cbc_decrypt_blocks, aes_decrypt_block,
    aes_decrypt_block_trace, inv_mix_columns, inv_mix_columns_detail, inv_shift_rows,
    inv_sub_bytes, pkcs7_unpad,
)
from backend.pipeline.encryption_aes import (
    MIX_MATRIX, InvalidCiphertextLength, PaddingError, add_round_key, aes_cbc_encrypt,
    aes_cbc_encrypt_blocks, aes_encrypt_block, aes_encrypt_block_trace, bytes_to_state,
    key_expansion, mix_columns, mix_columns_detail, pkcs7_pad, shift_rows, state_to_bytes,
    sub_bytes, sub_bytes_detail,
)
from backend.pipeline.gf256 import gf_inverse, gf_multiply, inverse_sbox_table, mul_table, sbox_table

H = bytes.fromhex
C1_KEY = H("000102030405060708090a0b0c0d0e0f")
C1_PT = H("00112233445566778899aabbccddeeff")
C1_CT = H("69c4e0d86a7b0430d8cdb78070b4c55a")
B_KEY = H("2b7e151628aed2a6abf7158809cf4f3c")
B_PT = H("3243f6a8885a308d313198a2e0370734")
B_CT = H("3925841d02dc09fbdc118597196a0b32")
SP_IV = H("000102030405060708090a0b0c0d0e0f")
SP_PT = [H("6bc1bee22e409f96e93d7e117393172a"), H("ae2d8a571e03ac9c9eb76fac45af8e51"),
         H("30c81c46a35ce411e5fbc1191a0a52ef"), H("f69f2445df4f9b17ad2b417be66c3710")]
SP_CBC_CT = [H("7649abac8119b246cee98e9b12e9197d"), H("5086cb9b507219ee95db113a917678b2"),
             H("73bed6b8e3c1743b7116e69e22229516"), H("3ff1caa1681fac09120eca307586e1a7")]


def rnd(rng, n):
    return bytes(rng.randrange(256) for _ in range(n))


def rstate(rng):
    return [[rng.randrange(256) for _ in range(4)] for _ in range(4)]


# ================================================================== CRITICAL GATE: FIPS-197 vector
def test_fips197_c1_encrypt_block():
    assert aes_encrypt_block(C1_PT, C1_KEY) == C1_CT
    assert aes_encrypt_block(C1_PT, C1_KEY).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"


def test_fips197_c1_decrypt_block():
    assert aes_decrypt_block(C1_CT, C1_KEY) == C1_PT


def test_fips197_appendix_b_vector():
    assert aes_encrypt_block(B_PT, B_KEY) == B_CT
    assert aes_decrypt_block(B_CT, B_KEY) == B_PT


def test_nist_sp800_38a_ecb_first_block():
    assert aes_encrypt_block(SP_PT[0], B_KEY).hex() == "3ad77bb40d7a3660a89ecaf32466ef97"


# ================================================================== key expansion
def test_key_expansion_shape_and_first_key():
    keys = key_expansion(C1_KEY)
    assert len(keys) == 11 and all(len(k) == 16 for k in keys)
    assert keys[0] == C1_KEY


def test_key_expansion_fips197_appendix_a1_words():
    keys = key_expansion(B_KEY)
    words = [k[i:i + 4].hex() for k in keys for i in range(0, 16, 4)]
    assert words[4:8] == ["a0fafe17", "88542cb1", "23a33939", "2a6c7605"]
    assert words[40:44] == ["d014f9a8", "c9ee2589", "e13f0cc8", "b6630ca6"]


def test_key_expansion_fips197_c1_round1_key():
    assert key_expansion(C1_KEY)[1].hex() == "d6aa74fdd2af72fadaa678f1d6ab76fe"


def test_rcon_values_derived_in_gf256():
    from backend.pipeline.encryption_aes import _rcon
    assert _rcon(10) == [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def test_key_expansion_zero_key_known_round1():
    # AES-128 all-zero key: w[4] = 62636363 (widely published), so round key 1 starts with 62636363.
    assert key_expansion(bytes(16))[1][:4].hex() == "62636363"


# ================================================================== state layout
def test_state_is_column_major():
    st = bytes_to_state(bytes(range(16)))
    assert st[0] == [0, 4, 8, 12] and st[1] == [1, 5, 9, 13] and st[3] == [3, 7, 11, 15]
    assert state_to_bytes(st) == bytes(range(16))


def test_state_round_trip_random():
    rng = random.Random(1)
    for _ in range(100):
        b = rnd(rng, 16)
        assert state_to_bytes(bytes_to_state(b)) == b


# ================================================================== round transformations
def test_fips197_c1_intermediate_states_round1():
    st = bytes_to_state(C1_PT)
    st = add_round_key(st, C1_KEY)
    assert state_to_bytes(st).hex() == "00102030405060708090a0b0c0d0e0f0"        # round[1].start
    st = sub_bytes(st)
    assert state_to_bytes(st).hex() == "63cab7040953d051cd60e0e7ba70e18c"        # round[1].s_box
    st = shift_rows(st)
    assert state_to_bytes(st).hex() == "6353e08c0960e104cd70b751bacad0e7"        # round[1].s_row
    st = mix_columns(st)
    assert state_to_bytes(st).hex() == "5f72641557f5bc92f7be3b291db9f91a"        # round[1].m_col
    st = add_round_key(st, key_expansion(C1_KEY)[1])
    assert state_to_bytes(st).hex() == "89d810e8855ace682d1843d8cb128fe4"        # round[2].start


def test_shift_rows_known():
    st = bytes_to_state(bytes(range(16)))                   # rows: [0 4 8 12] [1 5 9 13] ...
    out = shift_rows(st)
    assert out[0] == [0, 4, 8, 12] and out[1] == [5, 9, 13, 1]
    assert out[2] == [10, 14, 2, 6] and out[3] == [15, 3, 7, 11]


@pytest.mark.parametrize("col, expected", [
    ("db135345", "8e4da1bc"), ("f20a225c", "9fdc589d"), ("01010101", "01010101"),
    ("c6c6c6c6", "c6c6c6c6"), ("d4d4d4d5", "d5d5d7d6"), ("2d26314c", "4d7ebdf8"),
])
def test_mix_columns_known_columns(col, expected):
    c = H(col)
    st = [[c[r]] * 4 for r in range(4)]                     # every column equals c
    out = mix_columns(st)
    assert bytes(out[r][0] for r in range(4)).hex() == expected


def test_mix_columns_matrix_definition():
    assert MIX_MATRIX == ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))
    assert INV_MIX_MATRIX == ((14, 11, 13, 9), (9, 14, 11, 13), (13, 9, 14, 11), (11, 13, 9, 14))


def test_inverse_matrix_is_inverse_over_gf256():
    # M * M^-1 = I as 4x4 matrices over GF(2^8)
    for i in range(4):
        for j in range(4):
            acc = 0
            for k in range(4):
                acc ^= gf_multiply(MIX_MATRIX[i][k], INV_MIX_MATRIX[k][j])
            assert acc == (1 if i == j else 0)


def test_each_transformation_inverts_exactly():
    rng = random.Random(2)
    for _ in range(300):
        s = rstate(rng)
        assert inv_sub_bytes(sub_bytes(s)) == s
        assert sub_bytes(inv_sub_bytes(s)) == s
        assert inv_shift_rows(shift_rows(s)) == s
        assert shift_rows(inv_shift_rows(s)) == s
        assert inv_mix_columns(mix_columns(s)) == s
        assert mix_columns(inv_mix_columns(s)) == s


def test_add_round_key_is_self_inverse_and_uses_state_byte_order():
    rng = random.Random(3)
    s, rk = rstate(rng), rnd(rng, 16)
    assert add_round_key(add_round_key(s, rk), rk) == s
    out = add_round_key(bytes_to_state(bytes(16)), bytes(range(16)))
    assert out == bytes_to_state(bytes(range(16)))          # key byte i lands at state[i%4][i//4]


def test_transformations_do_not_mutate_their_input():
    s = bytes_to_state(bytes(range(16)))
    snapshot = [list(r) for r in s]
    for fn in (sub_bytes, shift_rows, mix_columns, inv_sub_bytes, inv_shift_rows, inv_mix_columns):
        fn(s)
    add_round_key(s, bytes(16))
    assert s == snapshot


def test_mix_columns_is_linear_over_gf2():
    rng = random.Random(4)
    for _ in range(100):
        a, b = rstate(rng), rstate(rng)
        x = [[a[r][c] ^ b[r][c] for c in range(4)] for r in range(4)]
        ma, mb, mx = mix_columns(a), mix_columns(b), mix_columns(x)
        assert mx == [[ma[r][c] ^ mb[r][c] for c in range(4)] for r in range(4)]


def test_mul_table_helpers_match_gf_multiply():
    for c in (2, 3, 9, 11, 13, 14):
        t = mul_table(c)
        assert isinstance(t, tuple) and len(t) == 256
        assert all(t[x] == gf_multiply(c, x) for x in range(256))
    assert mul_table(2) is mul_table(2)


def test_inverse_sbox_derivation():
    sb, inv = sbox_table(), inverse_sbox_table()
    assert all(inv[sb[x]] == x and sb[inv[x]] == x for x in range(256))
    # independent derivation: inverse affine map (rotl1 ^ rotl3 ^ rotl6 ^ 0x05) then field inversion
    def rotl(x, n):
        return ((x << n) | (x >> (8 - n))) & 0xFF
    for y in range(256):
        t = rotl(y, 1) ^ rotl(y, 3) ^ rotl(y, 6) ^ 0x05
        assert inv[y] == (gf_inverse(t) if t else 0)
    assert (inv[0x63], inv[0x7C], inv[0xED]) == (0x00, 0x01, 0x53)


def test_state_validation():
    for bad in ([[0] * 4] * 3, [[0] * 3] * 4, "abcd", [[256, 0, 0, 0]] + [[0] * 4] * 3,
                [[-1, 0, 0, 0]] + [[0] * 4] * 3, [[1.5, 0, 0, 0]] + [[0] * 4] * 3,
                [[True, 0, 0, 0]] + [[0] * 4] * 3):
        for fn in (sub_bytes, shift_rows, mix_columns, inv_sub_bytes, inv_shift_rows, inv_mix_columns):
            with pytest.raises(ValueError):
                fn(bad)
    with pytest.raises(ValueError):
        add_round_key(bytes_to_state(bytes(16)), bytes(15))


# ================================================================== block cipher properties
def test_block_round_trip_random_keys_and_blocks():
    rng = random.Random(5)
    for _ in range(400):
        key, block = rnd(rng, 16), rnd(rng, 16)
        ct = aes_encrypt_block(block, key)
        assert len(ct) == 16 and aes_decrypt_block(ct, key) == block


def test_encrypt_is_a_permutation_sample_and_key_dependent():
    rng = random.Random(6)
    key = rnd(rng, 16)
    outs = {aes_encrypt_block(rnd(rng, 16), key) for _ in range(200)}
    assert len(outs) == 200
    assert aes_encrypt_block(C1_PT, C1_KEY) != aes_encrypt_block(C1_PT, bytes([C1_KEY[0] ^ 1]) + C1_KEY[1:])


def test_avalanche_sanity_single_bit_change_flips_about_half_the_bits():
    # Experimental sanity check only (not a security proof): mean Hamming distance near 64 of 128.
    rng = random.Random(7)
    total, n = 0, 200
    for _ in range(n):
        key, block = rnd(rng, 16), bytearray(rnd(rng, 16))
        c1 = aes_encrypt_block(bytes(block), key)
        block[rng.randrange(16)] ^= 1 << rng.randrange(8)
        c2 = aes_encrypt_block(bytes(block), key)
        total += sum(bin(a ^ b).count("1") for a, b in zip(c1, c2))
    assert 56 <= total / n <= 72


def test_block_and_key_validation():
    for bad_key in (b"", bytes(15), bytes(17), bytes(24), bytes(32)):
        with pytest.raises(ValueError):
            aes_encrypt_block(C1_PT, bad_key)
        with pytest.raises(ValueError):
            aes_decrypt_block(C1_CT, bad_key)
        with pytest.raises(ValueError):
            key_expansion(bad_key)
    for bad_block in (b"", bytes(15), bytes(17)):
        with pytest.raises(ValueError):
            aes_encrypt_block(bad_block, C1_KEY)
        with pytest.raises(ValueError):
            aes_decrypt_block(bad_block, C1_KEY)
    for bad in ("abc", 123, None, [1] * 16):
        with pytest.raises(TypeError):
            aes_encrypt_block(bad, C1_KEY)
        with pytest.raises(TypeError):
            aes_encrypt_block(C1_PT, bad)
        with pytest.raises(TypeError):
            aes_decrypt_block(bad, C1_KEY)


def test_bytearray_inputs_accepted():
    assert aes_encrypt_block(bytearray(C1_PT), bytearray(C1_KEY)) == C1_CT


# ================================================================== traces
def test_encrypt_trace_structure_and_consistency():
    tr = aes_encrypt_block_trace(C1_PT, C1_KEY)
    steps = tr["steps"]
    assert len(steps) == 41                                         # input + 1 + 9*4 + 3
    assert tr["output_hex"] == C1_CT.hex() == steps[-1]["state_hex"]
    assert [s["step"] for s in steps[:6]] == ["Input", "AddRoundKey", "SubBytes", "ShiftRows", "MixColumns", "AddRoundKey"]
    assert [s["step"] for s in steps[-3:]] == ["SubBytes", "ShiftRows", "AddRoundKey"]   # no MixColumns last
    assert [s["round"] for s in steps][:2] == [0, 0] and steps[-1]["round"] == 10
    assert tr["round_keys_hex"] == [k.hex() for k in key_expansion(C1_KEY)]
    assert steps[1]["round_key_hex"] == C1_KEY.hex()
    # states in the trace equal the stand-alone transformations
    st = bytes_to_state(C1_PT)
    assert steps[0]["state"] == st
    st = add_round_key(st, C1_KEY)
    assert steps[1]["state"] == st
    st = sub_bytes(st); assert steps[2]["state"] == st
    st = shift_rows(st); assert steps[3]["state"] == st
    st = mix_columns(st); assert steps[4]["state"] == st
    assert steps[4]["state_hex"] == "5f72641557f5bc92f7be3b291db9f91a"


def test_gf_detail_reproduces_the_states():
    tr = aes_encrypt_block_trace(C1_PT, C1_KEY)
    for i, step in enumerate(tr["steps"]):
        if step["step"] == "MixColumns":
            before = tr["steps"][i - 1]["state"]
            assert step["detail"] == mix_columns_detail(before)
            for entry in step["detail"]:
                acc = 0
                for t in entry["terms"]:
                    assert t["product"] == gf_multiply(t["coef"], t["input"])
                    acc ^= t["product"]
                assert acc == entry["result"] == step["state"][entry["row"]][entry["column"]]
        if step["step"] == "SubBytes":
            before = tr["steps"][i - 1]["state"]
            for d in step["detail"]:
                assert d["output"] == step["state"][d["row"]][d["column"]]
                assert d["input"] == before[d["row"]][d["column"]]
                assert (d["inverse"] == 0) == (d["input"] == 0)
                if d["input"]:
                    assert gf_multiply(d["input"], d["inverse"]) == 1


def test_decrypt_trace_structure_and_consistency():
    tr = aes_decrypt_block_trace(C1_CT, C1_KEY)
    steps = tr["steps"]
    assert tr["output_hex"] == C1_PT.hex() == steps[-1]["state_hex"]
    assert steps[0]["step"] == "Input" and steps[0]["round"] == 10
    assert steps[-1]["step"] == "AddRoundKey" and steps[-1]["round"] == 0
    assert steps[-1]["round_key_hex"] == C1_KEY.hex()
    assert len(steps) == 2 + 3 * 10 + 9                             # input, rk10, 10x(3 steps), 9 InvMix
    # the decryption trace mirrors encryption: states are the encryption states in reverse order
    enc = aes_encrypt_block_trace(C1_PT, C1_KEY)["steps"]
    enc_hex = {s["state_hex"] for s in enc}
    for s in steps:
        assert s["state_hex"] in enc_hex
    for s in steps:
        if s["step"] == "InvMixColumns":
            for entry in s["detail"]:
                assert entry["result"] == s["state"][entry["row"]][entry["column"]]
    assert inv_mix_columns_detail(bytes_to_state(C1_PT)) is not None


def test_trace_does_not_change_ciphertext():
    rng = random.Random(8)
    for _ in range(30):
        key, block = rnd(rng, 16), rnd(rng, 16)
        tr = aes_encrypt_block_trace(block, key)
        assert tr["output_hex"] == aes_encrypt_block(block, key).hex()
        assert aes_decrypt_block_trace(bytes.fromhex(tr["output_hex"]), key)["output_hex"] == block.hex()


def test_trace_is_json_compatible():
    import json
    json.dumps(aes_encrypt_block_trace(C1_PT, C1_KEY))
    json.dumps(aes_decrypt_block_trace(C1_CT, C1_KEY))


# ================================================================== padding
@pytest.mark.parametrize("n, padded_len, pad_byte", [(0, 16, 16), (1, 16, 15), (15, 16, 1),
                                                      (16, 32, 16), (17, 32, 15), (31, 32, 1), (32, 48, 16)])
def test_pkcs7_pad_lengths(n, padded_len, pad_byte):
    out = pkcs7_pad(b"x" * n)
    assert len(out) == padded_len and out[-1] == pad_byte
    assert out[n:] == bytes([pad_byte]) * pad_byte
    assert pkcs7_unpad(out) == b"x" * n


def test_pkcs7_round_trip_all_lengths():
    for n in range(0, 70):
        data = bytes(range(n % 256)) if n < 256 else b""
        data = (data * 3)[:n]
        assert pkcs7_unpad(pkcs7_pad(data)) == data


@pytest.mark.parametrize("bad", [
    b"", b"A" * 15, b"A" * 17,                                           # wrong length
    b"A" * 15 + b"\x00",                                                 # pad byte 0
    b"A" * 15 + b"\x11",                                                 # pad byte 17 (> block size)
    b"A" * 14 + b"\x01\x02",                                             # claims 2 but bytes differ
    b"A" * 13 + b"\x03\x02\x03",                                         # inconsistent padding run
    b"A" * 16,                                                           # last byte 0x41 = 65
    bytes(16),                                                           # all zeros
])
def test_pkcs7_unpad_rejects_malformed(bad):
    with pytest.raises(PaddingError):
        pkcs7_unpad(bad)


def test_pkcs7_unpad_accepts_full_block_of_padding():
    assert pkcs7_unpad(b"\x10" * 16) == b""


def test_padding_error_hierarchy():
    assert issubclass(PaddingError, ValueError) and issubclass(InvalidCiphertextLength, ValueError)


# ================================================================== CBC
def test_nist_sp800_38a_cbc_vector():
    data = b"".join(SP_PT)
    ct = aes_cbc_encrypt(data, B_KEY, SP_IV)
    assert len(ct) == 80                                                  # 64 + one full padding block
    assert [ct[i:i + 16] for i in range(0, 64, 16)] == SP_CBC_CT
    assert aes_cbc_decrypt(ct, B_KEY, SP_IV) == data


def test_cbc_blocks_records_match_chaining_definition():
    recs = aes_cbc_encrypt_blocks(b"".join(SP_PT), B_KEY, SP_IV)
    assert len(recs) == 5
    prev = SP_IV
    for rec in recs:
        assert rec["cipher_input"] == bytes(a ^ b for a, b in zip(rec["plaintext"], prev))
        assert rec["ciphertext"] == aes_encrypt_block(rec["cipher_input"], B_KEY)
        prev = rec["ciphertext"]
    assert recs[0]["plaintext"] == SP_PT[0]
    dec = aes_cbc_decrypt_blocks(b"".join(r["ciphertext"] for r in recs), B_KEY, SP_IV)
    assert [d["plaintext"] for d in dec] == [r["plaintext"] for r in recs]
    assert all(d["decrypted"] == r["cipher_input"] for d, r in zip(dec, recs))


def test_cbc_round_trip_many_lengths():
    rng = random.Random(9)
    for n in list(range(0, 80)) + [127, 128, 129, 1000]:
        key, iv, data = rnd(rng, 16), rnd(rng, 16), rnd(rng, n)
        ct = aes_cbc_encrypt(data, key, iv)
        assert len(ct) == (n // 16 + 1) * 16
        assert aes_cbc_decrypt(ct, key, iv) == data


def test_cbc_empty_plaintext_is_one_block():
    ct = aes_cbc_encrypt(b"", C1_KEY, SP_IV)
    assert len(ct) == 16
    assert aes_cbc_decrypt(ct, C1_KEY, SP_IV) == b""
    assert ct == aes_encrypt_block(bytes(a ^ b for a, b in zip(b"\x10" * 16, SP_IV)), C1_KEY)


def test_cbc_unicode_round_trip():
    text = "héllo wörld 😀 日本語"
    ct = aes_cbc_encrypt(text.encode(), C1_KEY, SP_IV)
    assert aes_cbc_decrypt(ct, C1_KEY, SP_IV).decode() == text


def test_cbc_is_deterministic_and_iv_dependent():
    data = b"same message, different IV " * 3
    a = aes_cbc_encrypt(data, C1_KEY, bytes(16))
    assert a == aes_cbc_encrypt(data, C1_KEY, bytes(16))
    assert a != aes_cbc_encrypt(data, C1_KEY, bytes([1]) + bytes(15))


def test_cbc_hides_repeated_plaintext_blocks():
    ct = aes_cbc_encrypt(b"A" * 64, C1_KEY, SP_IV)
    blocks = [ct[i:i + 16] for i in range(0, len(ct), 16)]
    assert len(set(blocks)) == len(blocks)                                # ECB would repeat them


def test_cbc_bit_flip_propagation():
    # Flipping bit b in ciphertext block i garbles plaintext block i and flips bit b in block i+1.
    data = b"0123456789abcdef" * 4
    ct = bytearray(aes_cbc_encrypt(data, C1_KEY, SP_IV))
    ct[16 + 3] ^= 0x04                                                    # block 1, byte 3, bit 2
    recs = aes_cbc_decrypt_blocks(bytes(ct), C1_KEY, SP_IV)
    plain = [r["plaintext"] for r in recs]
    assert plain[0] == data[:16]                                          # earlier block untouched
    assert plain[1] != data[16:32]                                        # damaged block garbled
    assert plain[2][3] == data[32 + 3] ^ 0x04                             # exactly that bit flips next
    assert plain[2][:3] + plain[2][4:] == data[32:35] + data[36:48]
    assert plain[3] == data[48:64]                                        # later blocks untouched


def test_cbc_wrong_key_or_iv_does_not_recover_plaintext():
    data = b"secret message " * 3
    ct = aes_cbc_encrypt(data, C1_KEY, SP_IV)
    bad_key = bytes([C1_KEY[0] ^ 1]) + C1_KEY[1:]
    try:
        assert aes_cbc_decrypt(ct, bad_key, SP_IV) != data
    except PaddingError:
        pass
    wrong_iv = bytes([SP_IV[0] ^ 1]) + SP_IV[1:]
    first = aes_cbc_decrypt_blocks(ct, C1_KEY, wrong_iv)[0]["plaintext"]
    assert first != data[:16] and first[1:] == data[1:16]                 # wrong IV only damages block 0


def test_cbc_validation():
    for bad_iv in (b"", bytes(15), bytes(17)):
        with pytest.raises(ValueError):
            aes_cbc_encrypt(b"x", C1_KEY, bad_iv)
        with pytest.raises(ValueError):
            aes_cbc_decrypt(bytes(16), C1_KEY, bad_iv)
    for bad_key in (b"", bytes(15), bytes(32)):
        with pytest.raises(ValueError):
            aes_cbc_encrypt(b"x", bad_key, SP_IV)
        with pytest.raises(ValueError):
            aes_cbc_decrypt(bytes(16), bad_key, SP_IV)
    for bad in ("text", 5, None):
        with pytest.raises(TypeError):
            aes_cbc_encrypt(bad, C1_KEY, SP_IV)
        with pytest.raises(TypeError):
            aes_cbc_decrypt(bad, C1_KEY, SP_IV)
        with pytest.raises(TypeError):
            aes_cbc_encrypt(b"x", C1_KEY, bad)


@pytest.mark.parametrize("n", [0, 1, 15, 17, 31, 33])
def test_cbc_decrypt_rejects_bad_ciphertext_length(n):
    with pytest.raises(InvalidCiphertextLength):
        aes_cbc_decrypt(bytes(n), C1_KEY, SP_IV)


def test_cbc_decrypt_invalid_padding_after_decryption():
    # Encrypt a block whose plaintext ends with an invalid padding pattern using the raw primitive.
    bad_plain = b"A" * 15 + b"\x00"
    ct = aes_encrypt_block(bytes(a ^ b for a, b in zip(bad_plain, SP_IV)), C1_KEY)
    with pytest.raises(PaddingError):
        aes_cbc_decrypt(ct, C1_KEY, SP_IV)
    good_plain = b"A" * 15 + b"\x01"
    ct = aes_encrypt_block(bytes(a ^ b for a, b in zip(good_plain, SP_IV)), C1_KEY)
    assert aes_cbc_decrypt(ct, C1_KEY, SP_IV) == b"A" * 15


def test_cbc_decrypt_blocks_does_not_strip_or_validate_padding():
    bad_plain = b"A" * 15 + b"\x00"
    ct = aes_encrypt_block(bytes(a ^ b for a, b in zip(bad_plain, SP_IV)), C1_KEY)
    assert aes_cbc_decrypt_blocks(ct, C1_KEY, SP_IV)[0]["plaintext"] == bad_plain


def test_decryption_does_not_mutate_inputs():
    ct = bytearray(aes_cbc_encrypt(b"hello", C1_KEY, SP_IV))
    snapshot = bytes(ct)
    aes_cbc_decrypt(ct, C1_KEY, SP_IV)
    assert bytes(ct) == snapshot


def test_no_third_party_crypto_imports():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "backend" / "pipeline"
    for name in ("encryption_aes.py", "decryption_aes.py", "gf256.py"):
        src = (root / name).read_text()
        for banned in ("Crypto", "cryptography", "hashlib", "import zlib", "pycryptodome"):
            assert banned not in src, f"{name} mentions {banned}"


# ================================================================== gap-closing tests (found by mutation testing)
def test_every_addroundkey_step_labels_the_correct_round_key():
    for tr in (aes_encrypt_block_trace(C1_PT, C1_KEY), aes_decrypt_block_trace(C1_CT, C1_KEY)):
        checked = 0
        for step in tr["steps"]:
            if step["step"] == "AddRoundKey":
                assert step["round_key_hex"] == tr["round_keys_hex"][step["round"]]
                checked += 1
        assert checked == 11                                   # rounds 0..10 all covered


def test_encrypt_trace_addroundkey_rounds_are_0_to_10_in_order():
    rounds = [s["round"] for s in aes_encrypt_block_trace(C1_PT, C1_KEY)["steps"] if s["step"] == "AddRoundKey"]
    assert rounds == list(range(11))
    rounds = [s["round"] for s in aes_decrypt_block_trace(C1_CT, C1_KEY)["steps"] if s["step"] == "AddRoundKey"]
    assert rounds == list(range(10, -1, -1))                   # reverse order


def test_unpad_rejects_padding_longer_than_one_block():
    # 32 bytes ending in seventeen 0x11: internally "consistent" but a 17-byte pad is impossible.
    with pytest.raises(PaddingError, match="padding length byte 17"):
        pkcs7_unpad(b"A" * 15 + b"\x11" * 17)
    with pytest.raises(PaddingError, match="padding length byte 32"):
        pkcs7_unpad(b"\x20" * 32)


def test_unpad_zero_length_byte_has_specific_error():
    with pytest.raises(PaddingError, match="padding length byte 0"):
        pkcs7_unpad(b"A" * 15 + b"\x00")

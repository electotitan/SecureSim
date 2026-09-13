import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline import encryption_aes as aes
from pipeline import decryption_aes as aes_dec


def test_fips197_known_answer_vector():
    # FIPS-197 Appendix B / C.1 known-answer test:
    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
    expected_ciphertext = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")

    ciphertext = aes.aes_encrypt_block(plaintext, key)
    assert ciphertext == expected_ciphertext, (
        f"got {ciphertext.hex()}, expected {expected_ciphertext.hex()}"
    )


def test_fips197_decrypt_recovers_plaintext():
    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
    ciphertext = aes.aes_encrypt_block(plaintext, key)
    recovered = aes_dec.aes_decrypt_block(ciphertext, key)
    assert recovered == plaintext


def test_key_expansion_produces_11_round_keys_of_16_bytes():
    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    round_keys = aes.key_expansion(key)
    assert len(round_keys) == 11
    assert all(len(rk) == 16 for rk in round_keys)
    assert round_keys[0] == key  # round key 0 is the original key


def test_encrypt_decrypt_round_trip_random_blocks():
    import random
    random.seed(3)
    key = bytes(random.randint(0, 255) for _ in range(16))
    for _ in range(10):
        pt = bytes(random.randint(0, 255) for _ in range(16))
        ct = aes.aes_encrypt_block(pt, key)
        recovered = aes_dec.aes_decrypt_block(ct, key)
        assert recovered == pt


def test_cbc_round_trip_multi_block():
    import random
    random.seed(9)
    key = bytes(random.randint(0, 255) for _ in range(16))
    iv = bytes(random.randint(0, 255) for _ in range(16))
    for msg_len in [0, 1, 15, 16, 17, 33, 100]:
        data = bytes(random.randint(0, 255) for _ in range(msg_len))
        ct = aes.aes_cbc_encrypt(data, key, iv)
        assert len(ct) % 16 == 0
        recovered = aes_dec.aes_cbc_decrypt(ct, key, iv)
        assert recovered == data


def test_cbc_identical_blocks_produce_different_ciphertext():
    key = bytes(range(16))
    iv = bytes(range(16, 32))
    data = b"A" * 16 + b"A" * 16  # two identical plaintext blocks
    ct = aes.aes_cbc_encrypt(data, key, iv)
    block1, block2 = ct[0:16], ct[16:32]
    assert block1 != block2  # CBC chaining must break the ECB pattern leak


def test_state_bytes_round_trip():
    data = bytes(range(16))
    state = aes.bytes_to_state(data)
    assert aes.state_to_bytes(state) == data


def test_mix_columns_and_inverse_are_mutual_inverses():
    import random
    random.seed(5)
    state = [[random.randint(0, 255) for _ in range(4)] for _ in range(4)]
    mixed = aes.mix_columns(state)
    restored = aes.inv_mix_columns(mixed)
    assert restored == state


def test_shift_rows_and_inverse_are_mutual_inverses():
    state = [[r * 4 + c for c in range(4)] for r in range(4)]
    shifted = aes.shift_rows(state)
    restored = aes.inv_shift_rows(shifted)
    assert restored == state

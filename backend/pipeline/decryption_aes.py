"""
decryption_aes.py — AES-128 inverse cipher.

Decryption runs the same four transformations in reverse order, using
each one's mathematical inverse (InvSubBytes via the inverse S-box,
InvShiftRows via right-shifts, InvMixColumns via the inverse GF(2^8)
matrix {0e,0b,0d,09}, and AddRoundKey which is its own inverse since
XOR is an involution: A XOR K XOR K = A), and consumes round keys in
reverse order (last round key first). This mirrors the encrypt
schedule exactly, which is what makes AES a well-defined, invertible
permutation for each fixed key -- a requirement for any usable cipher.
"""

from __future__ import annotations
from . import gf256
from .encryption_aes import (
    NUM_ROUNDS,
    BLOCK_SIZE,
    bytes_to_state,
    state_to_bytes,
    inv_sub_bytes,
    inv_shift_rows,
    inv_mix_columns,
    add_round_key,
    key_expansion,
    pkcs7_unpad,
)


def aes_decrypt_block(ciphertext: bytes, key: bytes) -> bytes:
    """
    Decrypt one 16-byte block under AES-128, the exact mirror image of
    aes_encrypt_block:
        AddRoundKey(round 10)
        9 x [InvShiftRows, InvSubBytes, AddRoundKey, InvMixColumns]
        1 x [InvShiftRows, InvSubBytes, AddRoundKey]   (no InvMixColumns)
    """
    if len(ciphertext) != BLOCK_SIZE:
        raise ValueError("aes_decrypt_block requires exactly 16 bytes of ciphertext")
    round_keys = key_expansion(key)

    state = bytes_to_state(ciphertext)
    state = add_round_key(state, round_keys[NUM_ROUNDS])

    for rnd in range(NUM_ROUNDS - 1, 0, -1):
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        state = add_round_key(state, round_keys[rnd])
        state = inv_mix_columns(state)

    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    state = add_round_key(state, round_keys[0])

    return state_to_bytes(state)


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def aes_cbc_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """
    CBC decryption: P_i = AES_Decrypt(C_i) XOR C_{i-1} (C_0 = IV).
    Strips PKCS#7 padding after all blocks are decrypted.
    """
    if len(ciphertext) % BLOCK_SIZE != 0:
        raise ValueError("ciphertext length must be a multiple of the block size")
    plaintext = bytearray()
    prev = iv
    for i in range(0, len(ciphertext), BLOCK_SIZE):
        block = ciphertext[i:i + BLOCK_SIZE]
        dec = aes_decrypt_block(block, key)
        plaintext.extend(_xor_bytes(dec, prev))
        prev = block
    return pkcs7_unpad(bytes(plaintext))

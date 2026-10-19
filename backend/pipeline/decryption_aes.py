"""Stage 11 (AES part): AES-128 decryption (FIPS-197 inverse cipher), CBC mode and PKCS#7 unpadding.

EDUCATIONAL SIMULATOR ONLY (see ``encryption_aes``).  Same byte-order convention as encryption.

Inverse transformations
-----------------------
* InvShiftRows: row r rotated RIGHT by r.
* InvSubBytes: the inverse S-box (inverse permutation of the derived S-box).
* InvMixColumns: multiply each column by [14 11 13 9; 9 14 11 13; 13 9 14 11; 11 13 9 14] over GF(2^8)
  (the inverse of the MixColumns matrix; verified by tests).
* AddRoundKey is its own inverse (XOR); round keys are applied in REVERSE order (10 ... 0).

Inverse cipher: AddRoundKey(rk10); for r = 9..1: InvShiftRows, InvSubBytes, AddRoundKey(rk_r),
InvMixColumns; then InvShiftRows, InvSubBytes, AddRoundKey(rk0).

CBC decryption: P_i = D_K(C_i) XOR C_{i-1}, C_0 = IV; then the PKCS#7 padding is validated and removed.
Errors: ``InvalidCiphertextLength`` (empty / not a multiple of 16) and ``PaddingError`` (bad padding)
are distinct from argument errors.  CBC gives no authentication: a wrong key or a tampered ciphertext
usually yields garbage or a padding error, and cannot be told apart reliably.
"""
from __future__ import annotations

from typing import Any, Sequence

from backend.pipeline.encryption_aes import (
    BLOCK_SIZE, MIX_MATRIX, ROUNDS, InvalidCiphertextLength, PaddingError, State, _add_round_key,
    _as_bytes, _check_state, _record, _validate_block, _validate_iv, _validate_key, _xor16,
    bytes_to_state, key_expansion, state_to_bytes,
)
from backend.pipeline.gf256 import gf_multiply, inverse_sbox_table, mul_table

INV_MIX_MATRIX = ((14, 11, 13, 9), (9, 14, 11, 13), (13, 9, 14, 11), (11, 13, 9, 14))


# --------------------------------------------------------------------------- inverse transformations
def _inv_sub_bytes(state: State) -> State:
    inv = inverse_sbox_table()
    return [[inv[x] for x in row] for row in state]


def _inv_shift_rows(state: State) -> State:
    return [list(row[-r:]) + list(row[:-r]) if r else list(row) for r, row in enumerate(state)]


def _inv_mix_columns(state: State) -> State:
    t9, t11, t13, t14 = mul_table(9), mul_table(11), mul_table(13), mul_table(14)
    out = [[0] * 4 for _ in range(4)]
    for c in range(4):
        a0, a1, a2, a3 = state[0][c], state[1][c], state[2][c], state[3][c]
        out[0][c] = t14[a0] ^ t11[a1] ^ t13[a2] ^ t9[a3]
        out[1][c] = t9[a0] ^ t14[a1] ^ t11[a2] ^ t13[a3]
        out[2][c] = t13[a0] ^ t9[a1] ^ t14[a2] ^ t11[a3]
        out[3][c] = t11[a0] ^ t13[a1] ^ t9[a2] ^ t14[a3]
    return out


def inv_sub_bytes(state: State) -> State:
    """InvSubBytes: apply the inverse S-box to every byte."""
    _check_state(state)
    return _inv_sub_bytes(state)


def inv_shift_rows(state: State) -> State:
    """InvShiftRows: rotate row r right by r positions."""
    _check_state(state)
    return _inv_shift_rows(state)


def inv_mix_columns(state: State) -> State:
    """InvMixColumns: multiply every column by the inverse circulant matrix over GF(2^8)."""
    _check_state(state)
    return _inv_mix_columns(state)


# --------------------------------------------------------------------------- block cipher
def _decrypt_block(block: bytes, round_keys: Sequence[bytes]) -> bytes:
    state = _add_round_key(bytes_to_state(block), round_keys[ROUNDS])
    for r in range(ROUNDS - 1, 0, -1):
        state = _add_round_key(_inv_sub_bytes(_inv_shift_rows(state)), round_keys[r])
        state = _inv_mix_columns(state)
    state = _add_round_key(_inv_sub_bytes(_inv_shift_rows(state)), round_keys[0])
    return state_to_bytes(state)


def aes_decrypt_block(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt one 16-byte block with AES-128 (inverse of ``aes_encrypt_block``)."""
    block = _validate_block(ciphertext, "ciphertext")
    return _decrypt_block(block, key_expansion(key))


def inv_mix_columns_detail(state: State) -> list[dict[str, Any]]:
    """GF(2^8) arithmetic behind InvMixColumns for every output byte (uses gf_multiply directly)."""
    _check_state(state)
    detail = []
    for c in range(4):
        for r in range(4):
            terms, acc = [], 0
            for k in range(4):
                coef, inp = INV_MIX_MATRIX[r][k], state[k][c]
                prod = gf_multiply(coef, inp)
                terms.append({"coef": coef, "input": inp, "product": prod})
                acc ^= prod
            detail.append({"column": c, "row": r, "terms": terms, "result": acc})
    return detail


def aes_decrypt_block_trace(ciphertext: bytes, key: bytes) -> dict[str, Any]:
    """Decrypt one block, recording the state after every inverse transformation.

    Round numbers count DOWN (10 ... 0) so they match the encryption round whose effect is undone.
    The trace contains round keys derived from the key: treat it as secret material.
    """
    block = _validate_block(ciphertext, "ciphertext")
    round_keys = key_expansion(key)
    state = bytes_to_state(block)
    steps = [_record(ROUNDS, "Input", state)]
    state = _add_round_key(state, round_keys[ROUNDS])
    steps.append(_record(ROUNDS, "AddRoundKey", state, round_key_hex=round_keys[ROUNDS].hex()))
    for r in range(ROUNDS - 1, -1, -1):
        state = _inv_shift_rows(state)
        steps.append(_record(r, "InvShiftRows", state))
        state = _inv_sub_bytes(state)
        steps.append(_record(r, "InvSubBytes", state))
        state = _add_round_key(state, round_keys[r])
        steps.append(_record(r, "AddRoundKey", state, round_key_hex=round_keys[r].hex()))
        if r > 0:
            steps.append(_record(r, "InvMixColumns", _inv_mix_columns(state), detail=inv_mix_columns_detail(state)))
            state = _inv_mix_columns(state)
    return {
        "direction": "decrypt",
        "input_hex": block.hex(),
        "output_hex": state_to_bytes(state).hex(),
        "round_keys_hex": [rk.hex() for rk in round_keys],
        "byte_order": "column-major: state[r][c] = block[r + 4c]",
        "steps": steps,
    }


# --------------------------------------------------------------------------- padding and CBC
def pkcs7_unpad(data: bytes) -> bytes:
    """Validate and strip PKCS#7 padding; raises ``PaddingError`` if it is malformed."""
    data = _as_bytes(data, "data")
    if not data or len(data) % BLOCK_SIZE:
        raise PaddingError("padded data must be a non-empty multiple of the block size")
    n = data[-1]
    if not 1 <= n <= BLOCK_SIZE:
        raise PaddingError(f"invalid padding length byte {n}")
    if data[-n:] != bytes([n]) * n:
        raise PaddingError("padding bytes are inconsistent")
    return data[:-n]


def aes_cbc_decrypt_blocks(data: bytes, key: bytes, iv: bytes) -> list[dict[str, bytes]]:
    """CBC-decrypt WITHOUT removing padding; one record per block: ``ciphertext`` (C_i),
    ``decrypted`` (D_K(C_i)) and ``plaintext`` (D_K(C_i) XOR C_{i-1}, still padded)."""
    data = _as_bytes(data, "data")
    key = _validate_key(key)
    iv = _validate_iv(iv)
    if not data or len(data) % BLOCK_SIZE:
        raise InvalidCiphertextLength(
            f"ciphertext length must be a positive multiple of {BLOCK_SIZE}, got {len(data)}")
    round_keys = key_expansion(key)
    prev, records = iv, []
    for i in range(0, len(data), BLOCK_SIZE):
        block = data[i:i + BLOCK_SIZE]
        decrypted = _decrypt_block(block, round_keys)
        records.append({"ciphertext": block, "decrypted": decrypted, "plaintext": _xor16(decrypted, prev)})
        prev = block
    return records


def aes_cbc_decrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    """AES-128-CBC decryption with PKCS#7 validation (inverse of ``aes_cbc_encrypt``)."""
    records = aes_cbc_decrypt_blocks(data, key, iv)
    return pkcs7_unpad(b"".join(rec["plaintext"] for rec in records))

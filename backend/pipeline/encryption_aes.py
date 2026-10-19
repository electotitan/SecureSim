"""Stage 4B: AES-128 encryption (FIPS-197) with CBC mode and PKCS#7 padding, from first principles.

EDUCATIONAL SIMULATOR ONLY. This code is not constant-time, CBC provides no integrity, and
padding errors are reported distinctly (a padding-oracle risk).  Never use it to protect real data.

Byte-order convention (used everywhere, also by the visualisations)
-------------------------------------------------------------------
A 16-byte block b[0..15] is loaded into the 4x4 state COLUMN BY COLUMN:

        state[r][c] = b[r + 4*c]          (state is a list of 4 rows, each a list of 4 ints)

so bytes 0..3 form column 0, etc.  Round keys use the same order: round key r is the four
32-bit words w[4r], w[4r+1], w[4r+2], w[4r+3] concatenated, and AddRoundKey XORs byte i of the
round key into state[i % 4][i // 4].

Mathematics
-----------
* SubBytes: S(a) = Affine(a^-1) in GF(2^8) (table derived in ``gf256``; never hard-coded).
* ShiftRows: row r is rotated left by r positions.
* MixColumns: each column is multiplied by the circulant matrix [2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2]
  over GF(2^8) (equivalently by a(x) = 3x^3 + x^2 + x + 2 modulo x^4 + 1).
* AddRoundKey: XOR with the round key (addition in GF(2^8)).
* Key expansion (AES-128): w[i] = w[i-4] XOR T(w[i-1]) where, for i divisible by 4,
  T = SubWord(RotWord(.)) XOR Rcon[i/4] and otherwise T = identity.  Rcon[j] = x^(j-1) in GF(2^8),
  computed here by repeated multiplication by 2.
* 10 rounds: initial AddRoundKey, 9 full rounds (Sub, Shift, Mix, AddKey), final round without MixColumns.

CBC mode: C_0 = IV, C_i = E_K(P_i XOR C_{i-1}).  PKCS#7 padding ALWAYS appends n bytes of value n
(1 <= n <= 16); a block-aligned message therefore gets a full extra block, which keeps unpadding
unambiguous.  An empty message encrypts to exactly one block.
"""
from __future__ import annotations

from typing import Any, Sequence

from backend.pipeline.gf256 import gf_inverse, gf_multiply, mul_table, sbox_table

BLOCK_SIZE = 16
KEY_SIZE = 16
ROUNDS = 10

State = list[list[int]]


class AESError(ValueError):
    """Base class for data-dependent AES failures (as opposed to bad arguments)."""


class InvalidCiphertextLength(AESError):
    """Ciphertext is empty or not a multiple of the block size."""


class PaddingError(AESError):
    """PKCS#7 padding is malformed."""


# --------------------------------------------------------------------------- validation
def _as_bytes(value: Any, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes")
    return bytes(value)


def _validate_key(key: Any) -> bytes:
    key = _as_bytes(key, "key")
    if len(key) != KEY_SIZE:
        raise ValueError(f"AES-128 key must be exactly {KEY_SIZE} bytes, got {len(key)}")
    return key


def _validate_iv(iv: Any) -> bytes:
    iv = _as_bytes(iv, "iv")
    if len(iv) != BLOCK_SIZE:
        raise ValueError(f"IV must be exactly {BLOCK_SIZE} bytes, got {len(iv)}")
    return iv


def _validate_block(block: Any, name: str = "block") -> bytes:
    block = _as_bytes(block, name)
    if len(block) != BLOCK_SIZE:
        raise ValueError(f"{name} must be exactly {BLOCK_SIZE} bytes, got {len(block)}")
    return block


def _check_state(state: Any) -> None:
    if (not isinstance(state, (list, tuple)) or len(state) != 4
            or any(not isinstance(r, (list, tuple)) or len(r) != 4 for r in state)):
        raise ValueError("state must be a 4x4 matrix")
    for row in state:
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255:
                raise ValueError("state entries must be integers in 0..255")


# --------------------------------------------------------------------------- state conversion
def bytes_to_state(block: bytes) -> State:
    """Column-major load: state[r][c] = block[r + 4c]."""
    block = _validate_block(block)
    return [[block[r + 4 * c] for c in range(4)] for r in range(4)]


def state_to_bytes(state: State) -> bytes:
    """Inverse of :func:`bytes_to_state`."""
    _check_state(state)
    return bytes(state[r][c] for c in range(4) for r in range(4))


# --------------------------------------------------------------------------- round transformations
def _sub_bytes(state: State) -> State:
    sb = sbox_table()
    return [[sb[x] for x in row] for row in state]


def _shift_rows(state: State) -> State:
    return [list(row[r:]) + list(row[:r]) for r, row in enumerate(state)]


def _mix_columns(state: State) -> State:
    t2, t3 = mul_table(2), mul_table(3)
    out = [[0] * 4 for _ in range(4)]
    for c in range(4):
        a0, a1, a2, a3 = state[0][c], state[1][c], state[2][c], state[3][c]
        out[0][c] = t2[a0] ^ t3[a1] ^ a2 ^ a3
        out[1][c] = a0 ^ t2[a1] ^ t3[a2] ^ a3
        out[2][c] = a0 ^ a1 ^ t2[a2] ^ t3[a3]
        out[3][c] = t3[a0] ^ a1 ^ a2 ^ t2[a3]
    return out


def _add_round_key(state: State, round_key: bytes) -> State:
    return [[state[r][c] ^ round_key[r + 4 * c] for c in range(4)] for r in range(4)]


def sub_bytes(state: State) -> State:
    """SubBytes: apply the S-box to every state byte."""
    _check_state(state)
    return _sub_bytes(state)


def shift_rows(state: State) -> State:
    """ShiftRows: rotate row r left by r positions."""
    _check_state(state)
    return _shift_rows(state)


def mix_columns(state: State) -> State:
    """MixColumns: multiply every column by the fixed circulant matrix over GF(2^8)."""
    _check_state(state)
    return _mix_columns(state)


def add_round_key(state: State, round_key: bytes) -> State:
    """AddRoundKey: XOR the 16-byte round key into the state (same byte order as the state)."""
    _check_state(state)
    return _add_round_key(state, _validate_block(round_key, "round_key"))


# --------------------------------------------------------------------------- key schedule
def _rcon(count: int) -> list[int]:
    """Rcon[j] = x^(j-1): 01, 02, 04, ... by repeated multiplication by 2 in GF(2^8)."""
    values = [1]
    while len(values) < count:
        values.append(gf_multiply(values[-1], 2))
    return values


def key_expansion(key: bytes) -> list[bytes]:
    """Expand a 16-byte key into 11 round keys of 16 bytes each (FIPS-197 section 5.2)."""
    key = _validate_key(key)
    sb = sbox_table()
    rcon = _rcon(ROUNDS)
    words = [list(key[4 * i:4 * i + 4]) for i in range(4)]
    for i in range(4, 4 * (ROUNDS + 1)):
        temp = list(words[i - 1])
        if i % 4 == 0:
            temp = temp[1:] + temp[:1]                    # RotWord
            temp = [sb[b] for b in temp]                  # SubWord
            temp[0] ^= rcon[i // 4 - 1]
        words.append([a ^ b for a, b in zip(words[i - 4], temp)])
    return [bytes(b for w in words[4 * r:4 * r + 4] for b in w) for r in range(ROUNDS + 1)]


# --------------------------------------------------------------------------- block cipher
def _encrypt_block(block: bytes, round_keys: Sequence[bytes]) -> bytes:
    state = _add_round_key(bytes_to_state(block), round_keys[0])
    for r in range(1, ROUNDS):
        state = _add_round_key(_mix_columns(_shift_rows(_sub_bytes(state))), round_keys[r])
    state = _add_round_key(_shift_rows(_sub_bytes(state)), round_keys[ROUNDS])
    return state_to_bytes(state)


def aes_encrypt_block(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt one 16-byte block with AES-128 (ECB primitive)."""
    block = _validate_block(plaintext, "plaintext")
    return _encrypt_block(block, key_expansion(key))


# --------------------------------------------------------------------------- visualisation data
MIX_MATRIX = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))


def mix_columns_detail(state: State) -> list[dict[str, Any]]:
    """GF(2^8) arithmetic behind MixColumns for every output byte (uses gf_multiply directly).

    Entry per column c and row r: out[r][c] = XOR_k  MIX_MATRIX[r][k] * state[k][c].
    """
    _check_state(state)
    detail = []
    for c in range(4):
        for r in range(4):
            terms, acc = [], 0
            for k in range(4):
                coef, inp = MIX_MATRIX[r][k], state[k][c]
                prod = gf_multiply(coef, inp)
                terms.append({"coef": coef, "input": inp, "product": prod})
                acc ^= prod
            detail.append({"column": c, "row": r, "terms": terms, "result": acc})
    return detail


def sub_bytes_detail(state: State) -> list[dict[str, int]]:
    """For every state byte: input a, field inverse a^-1 (0 for 0) and S(a) = Affine(a^-1)."""
    _check_state(state)
    sb = sbox_table()
    return [{"row": r, "column": c, "input": state[r][c],
             "inverse": gf_inverse(state[r][c]) if state[r][c] else 0, "output": sb[state[r][c]]}
            for r in range(4) for c in range(4)]


def _record(round_no: int, step: str, state: State, **extra: Any) -> dict[str, Any]:
    return {"round": round_no, "step": step, "state": [list(r) for r in state],
            "state_hex": state_to_bytes(state).hex(), **extra}


def aes_encrypt_block_trace(plaintext: bytes, key: bytes) -> dict[str, Any]:
    """Encrypt one block and record the state after every transformation (JSON-compatible).

    The trace contains the round keys (derived from the key): required to show AddRoundKey,
    but it must be treated as secret material by callers.
    """
    block = _validate_block(plaintext, "plaintext")
    round_keys = key_expansion(key)
    state = bytes_to_state(block)
    steps = [_record(0, "Input", state)]
    state = _add_round_key(state, round_keys[0])
    steps.append(_record(0, "AddRoundKey", state, round_key_hex=round_keys[0].hex()))
    for r in range(1, ROUNDS + 1):
        steps.append(_record(r, "SubBytes", _sub_bytes(state), detail=sub_bytes_detail(state)))
        state = _sub_bytes(state)
        state = _shift_rows(state)
        steps.append(_record(r, "ShiftRows", state))
        if r < ROUNDS:
            steps.append(_record(r, "MixColumns", _mix_columns(state), detail=mix_columns_detail(state)))
            state = _mix_columns(state)
        state = _add_round_key(state, round_keys[r])
        steps.append(_record(r, "AddRoundKey", state, round_key_hex=round_keys[r].hex()))
    return {
        "direction": "encrypt",
        "input_hex": block.hex(),
        "output_hex": state_to_bytes(state).hex(),
        "round_keys_hex": [rk.hex() for rk in round_keys],
        "byte_order": "column-major: state[r][c] = block[r + 4c]",
        "steps": steps,
    }


# --------------------------------------------------------------------------- padding and CBC
def pkcs7_pad(data: bytes) -> bytes:
    """Append n bytes of value n, n in 1..16 (a full block when already aligned)."""
    data = _as_bytes(data, "data")
    n = BLOCK_SIZE - len(data) % BLOCK_SIZE
    return data + bytes([n]) * n


def _xor16(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def aes_cbc_encrypt_blocks(data: bytes, key: bytes, iv: bytes) -> list[dict[str, bytes]]:
    """PKCS#7-pad ``data`` and CBC-encrypt it, returning one record per block:
    ``plaintext`` (padded block P_i), ``cipher_input`` (P_i XOR C_{i-1}) and ``ciphertext`` (C_i).
    """
    data = _as_bytes(data, "data")
    key = _validate_key(key)
    iv = _validate_iv(iv)
    round_keys = key_expansion(key)
    padded = pkcs7_pad(data)
    prev, records = iv, []
    for i in range(0, len(padded), BLOCK_SIZE):
        block = padded[i:i + BLOCK_SIZE]
        cipher_input = _xor16(block, prev)
        prev = _encrypt_block(cipher_input, round_keys)
        records.append({"plaintext": block, "cipher_input": cipher_input, "ciphertext": prev})
    return records


def aes_cbc_encrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    """AES-128-CBC with PKCS#7 padding. The IV is NOT prepended to the output."""
    return b"".join(rec["ciphertext"] for rec in aes_cbc_encrypt_blocks(data, key, iv))

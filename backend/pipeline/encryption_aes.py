"""
encryption_aes.py — AES-128 block cipher, implemented from first principles.

Mathematical background
------------------------
AES treats a 16-byte block as a 4x4 matrix of GF(2^8) elements (the
"state"), filled column-major:

    [ s0  s4  s8  s12 ]
    [ s1  s5  s9  s13 ]
    [ s2  s6  s10 s14 ]
    [ s3  s7  s11 s15 ]

It then applies 10 rounds (for AES-128) of four transformations, each
grounded in GF(2^8) algebra (see gf256.py):

  * SubBytes   — apply the (nonlinear) S-box to every byte; this is the
                 only nonlinear step and is what gives AES its
                 resistance to linear/differential cryptanalysis.
  * ShiftRows  — cyclically left-shift row r by r positions; this
                 spreads a byte's influence horizontally across
                 columns over multiple rounds (diffusion).
  * MixColumns — treat each column as a degree-3 polynomial over
                 GF(2^8) and multiply it by the fixed polynomial
                 {03}x^3+{01}x^2+{01}x+{02} modulo x^4+1; this spreads
                 influence vertically within a column (more diffusion).
                 Skipped in the final round (by the AES spec) so
                 encryption remains invertible with a symmetric
                 structure.
  * AddRoundKey— XOR the state with that round's 16-byte round key
                 (this is where the secret key actually enters; every
                 other step is public and reversible without the key).

The round keys come from the Rijndael key schedule (key_expansion),
which recursively derives 11 round keys (round 0 = the original key)
using the S-box and a set of round constants Rcon = [x^(i-1) in
GF(2^8)] to break symmetry between otherwise-identical word positions.
"""

from __future__ import annotations
from . import gf256

NUM_ROUNDS = 10  # AES-128
BLOCK_SIZE = 16  # bytes


# ---------------------------------------------------------------------------
# Round constants (Rcon), used only in key expansion.
# Rcon[i] = (x^(i-1), 0, 0, 0) in GF(2^8), i.e. successive powers of x=0x02.
# ---------------------------------------------------------------------------
def _generate_rcon(num_words: int) -> list[int]:
    rcon = [0x00]  # rcon[0] unused (1-indexed convention below)
    val = 1
    for _ in range(num_words):
        rcon.append(val)
        val = gf256.gf_multiply(val, 0x02)
    return rcon


RCON = _generate_rcon(15)  # enough for AES-128/192/256 key schedules


# ---------------------------------------------------------------------------
# State <-> bytes helpers
# ---------------------------------------------------------------------------
def bytes_to_state(data: bytes) -> list[list[int]]:
    """16 bytes -> 4x4 state matrix, filled column-major (AES convention)."""
    state = [[0] * 4 for _ in range(4)]
    for i in range(16):
        state[i % 4][i // 4] = data[i]
    return state


def state_to_bytes(state: list[list[int]]) -> bytes:
    """4x4 state matrix -> 16 bytes, column-major."""
    out = bytearray(16)
    for i in range(16):
        out[i] = state[i % 4][i // 4]
    return bytes(out)


# ---------------------------------------------------------------------------
# The four round transformations
# ---------------------------------------------------------------------------
def sub_bytes(state: list[list[int]]) -> list[list[int]]:
    """Apply the S-box to every byte of the state (nonlinear substitution)."""
    return [[gf256.SBOX[state[r][c]] for c in range(4)] for r in range(4)]


def inv_sub_bytes(state: list[list[int]]) -> list[list[int]]:
    """Inverse of sub_bytes, using the inverse S-box."""
    return [[gf256.INV_SBOX[state[r][c]] for c in range(4)] for r in range(4)]


def shift_rows(state: list[list[int]]) -> list[list[int]]:
    """Cyclically left-shift row r by r positions (r = 0,1,2,3)."""
    new_state = [row[:] for row in state]
    for r in range(4):
        new_state[r] = state[r][r:] + state[r][:r]
    return new_state


def inv_shift_rows(state: list[list[int]]) -> list[list[int]]:
    """Inverse of shift_rows: cyclically right-shift row r by r positions."""
    new_state = [row[:] for row in state]
    for r in range(4):
        new_state[r] = state[r][-r:] + state[r][:-r] if r != 0 else state[r][:]
    return new_state


def _mix_single_column(col: list[int], matrix: list[list[int]]) -> list[int]:
    """Multiply a 4-byte column by a fixed 4x4 GF(2^8) matrix."""
    return [
        gf256.gf_multiply(matrix[r][0], col[0])
        ^ gf256.gf_multiply(matrix[r][1], col[1])
        ^ gf256.gf_multiply(matrix[r][2], col[2])
        ^ gf256.gf_multiply(matrix[r][3], col[3])
        for r in range(4)
    ]


# The MixColumns matrix, representing multiplication by {03}x^3+{01}x^2+{01}x+{02}
# modulo x^4+1, written out as circulant rows.
_MIX_MATRIX = [
    [0x02, 0x03, 0x01, 0x01],
    [0x01, 0x02, 0x03, 0x01],
    [0x01, 0x01, 0x02, 0x03],
    [0x03, 0x01, 0x01, 0x02],
]

# The inverse MixColumns matrix, representing multiplication by the inverse
# polynomial {0b}x^3+{0d}x^2+{09}x+{0e} modulo x^4+1.
_INV_MIX_MATRIX = [
    [0x0E, 0x0B, 0x0D, 0x09],
    [0x09, 0x0E, 0x0B, 0x0D],
    [0x0D, 0x09, 0x0E, 0x0B],
    [0x0B, 0x0D, 0x09, 0x0E],
]


def mix_columns(state: list[list[int]]) -> list[list[int]]:
    """Apply the MixColumns matrix to every column of the state."""
    new_state = [[0] * 4 for _ in range(4)]
    for c in range(4):
        col = [state[r][c] for r in range(4)]
        mixed = _mix_single_column(col, _MIX_MATRIX)
        for r in range(4):
            new_state[r][c] = mixed[r]
    return new_state


def inv_mix_columns(state: list[list[int]]) -> list[list[int]]:
    """Apply the inverse MixColumns matrix to every column of the state."""
    new_state = [[0] * 4 for _ in range(4)]
    for c in range(4):
        col = [state[r][c] for r in range(4)]
        mixed = _mix_single_column(col, _INV_MIX_MATRIX)
        for r in range(4):
            new_state[r][c] = mixed[r]
    return new_state


def add_round_key(state: list[list[int]], round_key: bytes) -> list[list[int]]:
    """XOR the state with the round key (both viewed as 16 bytes, column-major)."""
    rk_state = bytes_to_state(round_key)
    return [[state[r][c] ^ rk_state[r][c] for c in range(4)] for r in range(4)]


# ---------------------------------------------------------------------------
# Key expansion (Rijndael key schedule) -- AES-128: 4-word key -> 44 words
# grouped into 11 round keys of 4 words (16 bytes) each.
# ---------------------------------------------------------------------------
def _sub_word(word: list[int]) -> list[int]:
    return [gf256.SBOX[b] for b in word]


def _rot_word(word: list[int]) -> list[int]:
    return word[1:] + word[:1]


def key_expansion(key: bytes) -> list[bytes]:
    """
    Expand a 16-byte AES-128 key into 11 round keys (16 bytes each),
    following the Rijndael key schedule:

    The key is split into 4 words (4 bytes each): w[0..3].
    For i = 4..43:
        temp = w[i-1]
        if i % 4 == 0:
            temp = SubWord(RotWord(temp)) XOR (Rcon[i/4], 0, 0, 0)
        w[i] = w[i-4] XOR temp
    Every 4 consecutive words form one round key.
    """
    if len(key) != 16:
        raise ValueError("AES-128 key_expansion expects a 16-byte key")

    words: list[list[int]] = [list(key[4 * i:4 * i + 4]) for i in range(4)]

    total_words = 4 * (NUM_ROUNDS + 1)  # 44 words for AES-128
    for i in range(4, total_words):
        temp = words[i - 1][:]
        if i % 4 == 0:
            temp = _sub_word(_rot_word(temp))
            temp[0] ^= RCON[i // 4]
        new_word = [words[i - 4][j] ^ temp[j] for j in range(4)]
        words.append(new_word)

    round_keys = []
    for r in range(NUM_ROUNDS + 1):
        rk_words = words[4 * r: 4 * r + 4]
        rk_bytes = bytes(b for word in rk_words for b in word)
        round_keys.append(rk_bytes)
    return round_keys


# ---------------------------------------------------------------------------
# Single-block encryption / decryption
# ---------------------------------------------------------------------------
def aes_encrypt_block(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypt one 16-byte block under AES-128:
        AddRoundKey(round 0)
        9 x [SubBytes, ShiftRows, MixColumns, AddRoundKey]
        1 x [SubBytes, ShiftRows, AddRoundKey]   (final round, no MixColumns)
    """
    if len(plaintext) != BLOCK_SIZE:
        raise ValueError("aes_encrypt_block requires exactly 16 bytes of plaintext")
    round_keys = key_expansion(key)

    state = bytes_to_state(plaintext)
    state = add_round_key(state, round_keys[0])

    for rnd in range(1, NUM_ROUNDS):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[rnd])

    # Final round: no MixColumns
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[NUM_ROUNDS])

    return state_to_bytes(state)


def aes_encrypt_block_trace(plaintext: bytes, key: bytes) -> dict:
    """Same as aes_encrypt_block but also returns the state matrix after every
    transformation of every round, for frontend visualization."""
    round_keys = key_expansion(key)
    state = bytes_to_state(plaintext)
    trace = []

    state = add_round_key(state, round_keys[0])
    trace.append({"round": 0, "step": "AddRoundKey (initial)", "state": [row[:] for row in state]})

    for rnd in range(1, NUM_ROUNDS):
        state = sub_bytes(state)
        trace.append({"round": rnd, "step": "SubBytes", "state": [row[:] for row in state]})
        state = shift_rows(state)
        trace.append({"round": rnd, "step": "ShiftRows", "state": [row[:] for row in state]})
        state = mix_columns(state)
        trace.append({"round": rnd, "step": "MixColumns", "state": [row[:] for row in state]})
        state = add_round_key(state, round_keys[rnd])
        trace.append({"round": rnd, "step": "AddRoundKey", "state": [row[:] for row in state]})

    state = sub_bytes(state)
    trace.append({"round": NUM_ROUNDS, "step": "SubBytes", "state": [row[:] for row in state]})
    state = shift_rows(state)
    trace.append({"round": NUM_ROUNDS, "step": "ShiftRows", "state": [row[:] for row in state]})
    state = add_round_key(state, round_keys[NUM_ROUNDS])
    trace.append({"round": NUM_ROUNDS, "step": "AddRoundKey (final)", "state": [row[:] for row in state]})

    return {
        "ciphertext": state_to_bytes(state),
        "round_keys": round_keys,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# CBC mode: chains blocks together via XOR with the previous ciphertext block
# (or the IV, for the first block) so identical plaintext blocks do not
# produce identical ciphertext blocks.
# ---------------------------------------------------------------------------
def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def pkcs7_pad(data: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    """PKCS#7 padding: pad with N bytes each of value N, N = bytes needed to
    reach a full block (if data is already block-aligned, a full extra
    padding block of value block_size is added, so padding is always
    unambiguous to strip)."""
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len]) * pad_len


def pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        raise ValueError("cannot unpad empty data")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > BLOCK_SIZE or data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise ValueError("invalid PKCS#7 padding")
    return data[:-pad_len]


def aes_cbc_encrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    """
    CBC mode: C_0 = IV, C_i = AES_Encrypt(P_i XOR C_{i-1}).
    Chaining each block's *plaintext* with the previous *ciphertext*
    block (via XOR) before encrypting ensures identical plaintext
    blocks encrypt differently depending on context, hiding patterns
    that ECB mode would leak.
    """
    if len(iv) != BLOCK_SIZE:
        raise ValueError("IV must be 16 bytes")
    padded = pkcs7_pad(data)
    ciphertext = bytearray()
    prev = iv
    for i in range(0, len(padded), BLOCK_SIZE):
        block = padded[i:i + BLOCK_SIZE]
        xored = _xor_bytes(block, prev)
        enc = aes_encrypt_block(xored, key)
        ciphertext.extend(enc)
        prev = enc
    return bytes(ciphertext)

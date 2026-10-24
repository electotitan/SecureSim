"""Stage 7: CRC-32 error detection, derived from the generator polynomial.

CRC as polynomial division over GF(2)
-------------------------------------
Treat a message as a polynomial M(x) whose coefficients are its bits (here in TRANSMISSION order: byte by
byte, least significant bit of each byte first, which is the "reflected" convention).  Let G(x) be the
degree-32 generator polynomial of CRC-32 (IEEE 802.3, used by Ethernet, zlib, PNG, gzip):

    G(x) = x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
         = 0x1_04C11DB7           ("normal" form 0x04C11DB7 drops the implicit x^32 term)

The CRC is the remainder  R(x) = (M(x) * x^32) mod G(x),  with two conventions that make leading zeros and
trailing bytes matter:  the 32-bit register starts as all ones (init 0xFFFFFFFF) and the final register is
complemented (xorout 0xFFFFFFFF).  A sender appends the CRC; a receiver recomputes it over the received data and
compares.  Parameters (the "CRC-32/ISO-HDLC" catalogue entry):

    width 32   poly 0x04C11DB7   init 0xFFFFFFFF   refin true   refout true   xorout 0xFFFFFFFF
    check value for the ASCII string "123456789":  0xCBF43926

Three independent formulations (the tests require that they agree)
------------------------------------------------------------------
1. ``crc32_compute``  table-driven.  With the REFLECTED polynomial 0xEDB88320 (the bit-reversal of 0x04C11DB7) the
   register is shifted RIGHT; processing one byte is  reg = table[(reg ^ byte) & 0xFF] ^ (reg >> 8),  where
   table[i] is what the byte i becomes after eight shift-and-conditional-XOR steps.  The 256-entry table is
   DERIVED here from the polynomial (``generate_table``); it is not stored.
2. ``crc32_bitwise``  the same arithmetic one bit at a time, no table.
3. ``crc32_polynomial_division``  literal long division of the augmented message (message followed by 32 zero
   bits, first 32 bits inverted to model init = all ones) by G(x) in the non-reflected domain, then reflect the
   remainder and complement it.

Properties (each verified by tests) and limits (stated, not hidden)
-------------------------------------------------------------------
* Detects EVERY single-bit error and EVERY error burst of length <= 32 bits (an error polynomial of degree < 32
  can never be a nonzero multiple of G(x)).  Other error patterns escape detection with probability about
  2^-32 (roughly 2.3e-10) for "random" damage.  We make no claim about odd numbers of errors: G(x) has an odd
  number of terms, so it is NOT divisible by (x + 1).
* Detects but never CORRECTS: a mismatch says "something changed", not where or how.
* Linear (affine) over GF(2): crc(a ^ b ^ c) = crc(a) ^ crc(b) ^ crc(c) for equal-length a, b, c.  Hence it is
  NOT a cryptographic authentication mechanism: ``crc32_forge_suffix`` computes 4 bytes that force any chosen
  CRC value, so an attacker can alter a message and repair the checksum.  Use a MAC (HMAC, AEAD) for that.
* Residue: crc32(data + crc32(data) as 4 little-endian bytes) = 0x2144DF1C for every data.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

POLY_NORMAL = 0x04C11DB7
POLY_FULL = 0x104C11DB7
INIT = 0xFFFFFFFF
XOROUT = 0xFFFFFFFF
CHECK_INPUT = b"123456789"
CHECK_VALUE = 0xCBF43926
RESIDUE = 0x2144DF1C
MASK32 = 0xFFFFFFFF


def _reflect32(value: int) -> int:
    """Reverse the order of the 32 bits of ``value``."""
    return int(format(value & MASK32, "032b")[::-1], 2)


POLY_REFLECTED = _reflect32(POLY_NORMAL)
assert POLY_REFLECTED == 0xEDB88320          # the reflected polynomial is DERIVED from the normal form


# --------------------------------------------------------------------------- validation
def _as_bytes(data: Any, name: str = "data") -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes")
    return bytes(data)


def _check_u32(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if not 0 <= value <= MASK32:
        raise ValueError(f"{name} must be in the range 0..{MASK32} (a 32-bit unsigned value)")
    return value


# --------------------------------------------------------------------------- table derivation
def generate_table(reflected_poly: int = POLY_REFLECTED) -> tuple[int, ...]:
    """The 256-entry CRC table for a reflected polynomial.

    Entry i is the register obtained by loading the byte i and doing eight steps of: shift right by one, and if the
    bit shifted out was 1, XOR with the reflected polynomial.
    """
    _check_u32(reflected_poly, "reflected_poly")
    table = []
    for i in range(256):
        reg = i
        for _ in range(8):
            reg = (reg >> 1) ^ reflected_poly if reg & 1 else reg >> 1
        table.append(reg)
    return tuple(table)


_TABLE = generate_table()
# The top bytes of the entries are all distinct (a permutation of 0..255); ``crc32_forge_suffix`` relies on it.
_TOP_BYTE_TO_INDEX = {entry >> 24: i for i, entry in enumerate(_TABLE)}
assert len(_TOP_BYTE_TO_INDEX) == 256


def crc32_table() -> tuple[int, ...]:
    """The derived CRC-32 table (immutable)."""
    return _TABLE


# --------------------------------------------------------------------------- polynomial representation
def polynomial_terms(full_polynomial: int = POLY_FULL) -> list[int]:
    """Exponents of the nonzero terms of a GF(2) polynomial, highest first (0x104C11DB7 -> [32, 26, 23, ...])."""
    if isinstance(full_polynomial, bool) or not isinstance(full_polynomial, int) or full_polynomial <= 0:
        raise ValueError("polynomial must be a positive int")
    return [e for e in range(full_polynomial.bit_length() - 1, -1, -1) if (full_polynomial >> e) & 1]


def polynomial_expression(full_polynomial: int = POLY_FULL) -> str:
    """Human-readable form such as 'x^32 + x^26 + ... + x + 1'."""
    parts = []
    for e in polynomial_terms(full_polynomial):
        parts.append("1" if e == 0 else ("x" if e == 1 else f"x^{e}"))
    return " + ".join(parts)


# --------------------------------------------------------------------------- core computation
def crc32_state(data: bytes, state: int = INIT) -> int:
    """Run ``data`` through the register (table-driven) WITHOUT the final complement."""
    data = _as_bytes(data)
    _check_u32(state, "state")
    table = _TABLE
    for byte in data:
        state = table[(state ^ byte) & 0xFF] ^ (state >> 8)
    return state


def crc32_compute(data: bytes, previous: int = 0) -> int:
    """CRC-32 of ``data`` as an unsigned 32-bit int.

    ``previous`` continues a running checksum: crc32_compute(b, crc32_compute(a)) == crc32_compute(a + b).
    """
    _check_u32(previous, "previous")
    return crc32_state(data, previous ^ XOROUT) ^ XOROUT


def crc32_verify(data: bytes, checksum: int) -> bool:
    """True iff the CRC-32 of ``data`` equals ``checksum``."""
    _check_u32(checksum, "checksum")
    return crc32_compute(data) == checksum


def crc32_bitwise(data: bytes) -> int:
    """Reference implementation: one bit at a time, no table."""
    data = _as_bytes(data)
    reg = INIT
    for byte in data:
        reg ^= byte
        for _ in range(8):
            reg = (reg >> 1) ^ POLY_REFLECTED if reg & 1 else reg >> 1
    return reg ^ XOROUT


def crc32_polynomial_division(data: bytes) -> int:
    """Reference implementation by long division in the NON-reflected domain (see the module docstring).

    The augmented message (message bits in transmission order, then 32 zero bits, with the first 32 bits inverted
    to model init = all ones) is divided by G(x) with a 32-bit remainder register; the remainder is bit-reflected
    and complemented.
    """
    data = _as_bytes(data)
    bits = [(byte >> k) & 1 for byte in data for k in range(8)] + [0] * 32
    for i in range(32):
        bits[i] ^= 1
    rem = 0
    for b in bits:
        top = (rem >> 31) & 1
        rem = ((rem << 1) | b) & MASK32
        if top:
            rem ^= POLY_NORMAL
    return _reflect32(rem) ^ XOROUT


# --------------------------------------------------------------------------- visualisation
def crc32_trace(data: bytes, max_byte_steps: int = 32, bit_steps_bytes: int = 2) -> dict[str, Any]:
    """Register values while computing the CRC (JSON-compatible; integers).

    ``byte_steps``: for each of the first ``max_byte_steps`` bytes: the register before, the byte, the table index
    (register XOR byte, low 8 bits), the table entry and the register after.  ``bit_steps`` (first
    ``bit_steps_bytes`` bytes only): the eight shift / conditional-XOR steps that produce the same register.
    """
    data = _as_bytes(data)
    for name, v in (("max_byte_steps", max_byte_steps), ("bit_steps_bytes", bit_steps_bytes)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"{name} must be a non-negative int")
    reg = INIT
    byte_steps, bit_steps = [], []
    for i, byte in enumerate(data):
        before = reg
        index = (reg ^ byte) & 0xFF
        reg = _TABLE[index] ^ (reg >> 8)
        if i < max_byte_steps:
            byte_steps.append({"index": i, "byte": byte, "register_before": before, "table_index": index,
                               "table_value": _TABLE[index], "register_after": reg})
        if i < bit_steps_bytes:
            r = before ^ byte
            steps = [{"step": 0, "lsb": None, "xor_polynomial": None, "register": r}]
            for k in range(1, 9):
                lsb = r & 1
                r = (r >> 1) ^ POLY_REFLECTED if lsb else r >> 1
                steps.append({"step": k, "lsb": lsb, "xor_polynomial": bool(lsb), "register": r})
            assert r == reg                                   # the eight bit steps equal the table step
            bit_steps.append({"byte_index": i, "byte": byte, "steps": steps})
    return {
        "polynomial_normal": POLY_NORMAL, "polynomial_reflected": POLY_REFLECTED, "polynomial_full": POLY_FULL,
        "polynomial_terms": polynomial_terms(), "polynomial_expression": polynomial_expression(),
        "initial_register": INIT, "byte_steps": byte_steps, "bit_steps": bit_steps,
        "steps_total": len(data), "steps_returned": len(byte_steps), "steps_truncated": len(byte_steps) < len(data),
        "final_register": reg, "final_xor": XOROUT, "checksum": reg ^ XOROUT,
    }


# --------------------------------------------------------------------------- error experiments
def flip_bits(data: bytes, positions: Sequence[int]) -> bytes:
    """Flip the given bits. Bit position p is bit (p % 8) of byte (p // 8), counting bit 0 as the LEAST
    significant bit: this is the order in which a reflected CRC "sees" (transmits) the bits.  Positions are
    applied as an XOR mask, so a position listed twice cancels itself."""
    data = bytearray(_as_bytes(data))
    for p in positions:
        if isinstance(p, bool) or not isinstance(p, int):
            raise TypeError("bit positions must be ints")
        if not 0 <= p < 8 * len(data):
            raise ValueError(f"bit position {p} is outside the data ({8 * len(data)} bits)")
        data[p // 8] ^= 1 << (p % 8)
    return bytes(data)


def error_span(positions: Sequence[int]) -> Optional[int]:
    """Distance from the first to the last flipped bit, inclusive (the burst length); None if nothing flipped."""
    ps = list(positions)
    return (max(ps) - min(ps) + 1) if ps else None


# --------------------------------------------------------------------------- forging (why CRC is not authentication)
def crc32_forge_suffix(data: bytes, target: int) -> bytes:
    """Four bytes that, appended to ``data``, make the CRC-32 equal ``target``.

    After four more bytes the register is  T4 = table[i3] ^ (table[i2] >> 8) ^ (table[i1] >> 16) ^ (table[i0] >> 24)
    (the old state has been shifted out completely), where i_k are the table indices used.  The top byte of each
    table entry is distinct, so the indices are determined one byte at a time from the wanted register
    (target ^ xorout); the four bytes then follow from the current register.
    """
    data = _as_bytes(data)
    target = _check_u32(target, "target")
    wanted = target ^ XOROUT
    idx = [0, 0, 0, 0]
    rest = wanted
    for k in (3, 2, 1, 0):                                   # i3 from the top byte, then i2, i1, i0
        i = _TOP_BYTE_TO_INDEX[(rest >> 24) & 0xFF]
        idx[k] = i
        rest = ((rest ^ _TABLE[i]) << 8) & MASK32            # strip table[i], expose the next byte on top
    reg = crc32_state(data)
    suffix = bytearray()
    for k in range(4):
        byte = (reg & 0xFF) ^ idx[k]
        suffix.append(byte)
        reg = _TABLE[idx[k]] ^ (reg >> 8)
    assert reg == wanted
    return bytes(suffix)

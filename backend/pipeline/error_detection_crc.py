"""
error_detection_crc.py — CRC-32, via polynomial long division over GF(2).

Mathematical background
------------------------
A Cyclic Redundancy Check treats the message as a polynomial M(x) with
coefficients in GF(2) (each bit is one coefficient), and computes the
remainder of M(x)*x^32 divided by a fixed degree-32 generator
polynomial G(x), using polynomial arithmetic over GF(2) (addition =
XOR, no carries -- the same field structure gf256.py uses for AES, but
here bit-wise on an arbitrarily long polynomial rather than byte-wise
on GF(2^8)).

Standard CRC-32 (the one used by Ethernet, zip, PNG, etc.) uses the
generator polynomial
    G(x) = x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11 + x^10
           + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
which corresponds to the bit pattern 0x04C11DB7 (the top x^32 term is
implicit -- G(x) has degree 32 but we only need its low 32 bits to
divide, exactly like a "monic" polynomial in ordinary division).

The received message R(x) is judged intact iff R(x) mod G(x) == 0. We
implement this directly via bit-by-bit polynomial long division (the
"shift and conditionally XOR" method), and additionally derive the
standard byte-oriented lookup table *from* that division routine
(rather than hardcoding Ethernet's published table), since the
table's 256 entries are themselves just "the CRC remainder you'd get
from dividing that one byte, pre-shifted" -- a classic speed
optimization that trades a one-time O(256*8) table build for O(1)
work per input byte thereafter.

Standard CRC-32 also reflects input/output bits and XORs with
0xFFFFFFFF at the start and end; we replicate that so our function
matches the widely-known published test vector "123456789" -> 0xCBF43926.
"""

from __future__ import annotations

CRC32_POLY = 0x04C11DB7  # standard CRC-32 generator polynomial (low 32 bits, top term implicit)


def _reflect(value: int, num_bits: int) -> int:
    """Reverse the bit order of the lowest `num_bits` bits of `value`."""
    result = 0
    for _ in range(num_bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _divide_byte_by_polynomial(byte_val: int, poly: int) -> int:
    """
    Core polynomial long division step for one byte: XOR the byte
    (shifted into the top of a 32-bit register) with the generator
    polynomial whenever the current leading bit is 1, shifting left
    each time -- exactly like ordinary long division, but XOR replaces
    subtraction because GF(2) subtraction and addition coincide.
    """
    reg = byte_val << 24
    for _ in range(8):
        if reg & 0x80000000:
            reg = ((reg << 1) & 0xFFFFFFFF) ^ poly
        else:
            reg = (reg << 1) & 0xFFFFFFFF
    return reg


def _build_crc32_table(poly: int = CRC32_POLY) -> list[int]:
    """
    Derive the standard byte-oriented CRC-32 lookup table by running
    the bit-level division on every possible input byte (0..255),
    using the *reflected* (bit-reversed) form of the polynomial, since
    standard CRC-32 processes bits least-significant-bit first.
    table[i] = the 32-bit division remainder produced by feeding byte i
    through the reflected-polynomial division circuit.
    """
    reflected_poly = _reflect(poly, 32)
    table = []
    for byte_val in range(256):
        crc = byte_val
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ reflected_poly
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFF)
    return table


CRC32_TABLE = _build_crc32_table()


def crc32_compute(data: bytes) -> int:
    """
    Standard CRC-32 (as used by Ethernet/zip/PNG): initialize register
    to 0xFFFFFFFF, process each byte via the derived table (table
    lookup XORed with the shifted register -- equivalent to, but much
    faster than, doing full bit-by-bit division per byte), then
    complement (XOR 0xFFFFFFFF) the final register.

    The initial/final XOR with all-ones and the LSB-first bit
    processing are historical conventions of the "CRC-32" standard
    (distinct from the "pure" MSB-first division CRC32_POLY alone
    would define) that make it robust to leading/trailing zero bytes.
    """
    crc = 0xFFFFFFFF
    for byte_val in data:
        table_index = (crc ^ byte_val) & 0xFF
        crc = (crc >> 8) ^ CRC32_TABLE[table_index]
    return crc ^ 0xFFFFFFFF


def crc32_verify(data: bytes, checksum: int) -> bool:
    """A message is judged intact iff recomputing CRC-32 over it reproduces `checksum`."""
    return crc32_compute(data) == checksum


def crc32_bitwise_reference(data: bytes) -> int:
    """
    A slower, more literal MSB-first bit-by-bit polynomial division,
    kept as an independent reference implementation to cross-check the
    table-driven crc32_compute where the two conventions coincide
    (used only in tests, not the hot path).
    """
    reg = 0xFFFFFFFF
    for byte_val in data:
        reg ^= _reflect(byte_val, 8) << 24
        for _ in range(8):
            if reg & 0x80000000:
                reg = ((reg << 1) & 0xFFFFFFFF) ^ CRC32_POLY
            else:
                reg = (reg << 1) & 0xFFFFFFFF
    return _reflect(reg, 32) ^ 0xFFFFFFFF

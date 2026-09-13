"""
binary_repr.py — Explicit text <-> binary conversion (Number Systems).

Every character is first encoded as UTF-8 bytes (the standard variable
-length byte encoding of Unicode code points), and every byte is then
written out as an explicit 8-bit binary string (base-2 positional
notation: byte = sum_{i=0}^{7} bit_i * 2^i, most-significant bit first).
This stage is deliberately "boring" mathematically (it's a base
conversion, radix 256 -> radix 2) but it is the literal substrate every
later stage operates on.
"""

from __future__ import annotations


def text_to_binary(text: str) -> str:
    """Encode text as UTF-8 bytes, then each byte as 8 bits (MSB first)."""
    data = text.encode("utf-8")
    return "".join(format(byte, "08b") for byte in data)


def binary_to_text(bits: str) -> str:
    """Inverse of text_to_binary: regroup into bytes, then UTF-8-decode."""
    if len(bits) % 8 != 0:
        raise ValueError(f"bit length {len(bits)} is not a multiple of 8")
    byte_values = bytearray()
    for i in range(0, len(bits), 8):
        byte_values.append(int(bits[i:i + 8], 2))
    return bytes(byte_values).decode("utf-8")


def visualize_bits(bits: str) -> list[dict]:
    """
    Group a bitstring into byte-sized chunks with metadata useful for a
    frontend byte-grid display: the bit pattern, its integer value, and
    (where printable) the corresponding ASCII character.
    """
    if len(bits) % 8 != 0:
        raise ValueError(f"bit length {len(bits)} is not a multiple of 8")
    grid = []
    for idx, i in enumerate(range(0, len(bits), 8)):
        byte_bits = bits[i:i + 8]
        value = int(byte_bits, 2)
        ascii_char = chr(value) if 32 <= value <= 126 else None
        grid.append({
            "index": idx,
            "bits": byte_bits,
            "value": value,
            "hex": format(value, "02x"),
            "ascii": ascii_char,
        })
    return grid


def bytes_to_bits(data: bytes) -> str:
    """General-purpose raw-bytes -> bitstring conversion (8 bits/byte, MSB
    first), used to hand arbitrary binary data -- not just UTF-8 text -- to
    bit-oriented stages such as the channel simulator and Hamming coding."""
    return "".join(format(b, "08b") for b in data)


def bits_to_bytes(bits: str) -> bytes:
    """Inverse of bytes_to_bits: regroup a bitstring into raw bytes (no
    UTF-8 decoding -- unlike binary_to_text, the result need not be valid
    text, e.g. it may be ciphertext or an error-correction codeword)."""
    if len(bits) % 8 != 0:
        raise ValueError(f"bit length {len(bits)} is not a multiple of 8")
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))

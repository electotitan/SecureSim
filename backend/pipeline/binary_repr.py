"""Stage 2: binary representation of text via UTF-8.

Two different things are easy to confuse:

* a *character-to-integer* mapping: Unicode assigns every character an abstract code point
  (e.g. 'e-acute' = U+00E9 = 233).  This is NOT a byte sequence.
* a *text encoding*: UTF-8 turns each code point into 1-4 bytes:

      code point range        byte pattern
      U+0000   .. U+007F      0xxxxxxx
      U+0080   .. U+07FF      110xxxxx 10xxxxxx
      U+0800   .. U+FFFF      1110xxxx 10xxxxxx 10xxxxxx
      U+10000  .. U+10FFFF    11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

  'e-acute' (233) becomes the two bytes 0xC3 0xA9, not the single byte 0xE9.

Bit string convention: a ``str`` of '0'/'1' characters, 8 per byte, most significant bit first.
Python's own ``str.encode`` / ``bytes.decode`` (strict) are used for the UTF-8 mapping itself;
the 8-bit expansion and all validation are implemented here.
"""
from __future__ import annotations

from typing import Any, Optional


def _validate_bits(bits: str, *, require_whole_bytes: bool = True) -> None:
    if not isinstance(bits, str):
        raise TypeError("bits must be a str")
    if set(bits) - {"0", "1"}:
        raise ValueError("binary input may contain only the characters '0' and '1'")
    if require_whole_bytes and len(bits) % 8 != 0:
        raise ValueError(f"bit length must be a multiple of 8 (complete bytes), got {len(bits)}")


def bytes_to_binary(data: bytes) -> str:
    """Each byte becomes exactly eight bits, most significant bit first."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    return "".join(format(b, "08b") for b in data)


def binary_to_bytes(bits: str) -> bytes:
    """Inverse of :func:`bytes_to_binary`; requires complete bytes."""
    _validate_bits(bits)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def text_to_binary(text: str) -> str:
    """UTF-8 encode ``text`` and expand every byte to eight bits."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("text contains characters that cannot be encoded as UTF-8 "
                         "(for example a lone surrogate)") from exc
    return bytes_to_binary(data)


def binary_to_text(bits: str) -> str:
    """Group bits into bytes and strictly decode them as UTF-8."""
    data = binary_to_bytes(bits)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"bytes are not valid UTF-8: {exc.reason} at byte {exc.start}") from exc


def _byte_role(value: int) -> str:
    if value < 0x80:
        return "ascii"
    if value >> 6 == 0b10:
        return "continuation"
    if value >> 5 == 0b110:
        return "lead-2"
    if value >> 4 == 0b1110:
        return "lead-3"
    if value >> 3 == 0b11110:
        return "lead-4"
    return "invalid"


def visualize_bits(bits: str) -> list[dict[str, Any]]:
    """Per-byte description for the byte grid.

    Each entry: index, binary (8 chars), bits (list of 8 ints, MSB first), hex, decimal,
    role (UTF-8 byte class from its bit pattern), character (the whole character this byte
    belongs to when the stream is valid UTF-8, else None), char_index, char_start (True for
    the first byte of a character).  Arbitrary byte streams (e.g. ciphertext) are accepted;
    they simply have character = None.
    """
    data = binary_to_bytes(bits)
    char_of: list[Optional[tuple[int, str, bool]]] = [None] * len(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is not None:
        pos = 0
        for char_index, ch in enumerate(text):
            for k in range(len(ch.encode("utf-8"))):
                char_of[pos + k] = (char_index, ch, k == 0)
            pos += len(ch.encode("utf-8"))

    result = []
    for i, value in enumerate(data):
        info = char_of[i]
        result.append({
            "index": i,
            "binary": format(value, "08b"),
            "bits": [int(c) for c in format(value, "08b")],
            "hex": format(value, "02x"),
            "decimal": value,
            "role": _byte_role(value),
            "character": info[1] if info else None,
            "char_index": info[0] if info else None,
            "char_start": info[2] if info else False,
        })
    return result


def describe_characters(text: str) -> list[dict[str, Any]]:
    """Contrast code point (abstract integer) with its UTF-8 bytes, per character."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    out = []
    for i, ch in enumerate(text):
        try:
            encoded = ch.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("text contains characters that cannot be encoded as UTF-8") from exc
        out.append({
            "index": i,
            "character": ch,
            "code_point": ord(ch),
            "code_point_hex": f"U+{ord(ch):04X}",
            "utf8_hex": encoded.hex(),
            "utf8_byte_count": len(encoded),
        })
    return out

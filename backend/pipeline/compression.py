"""Stage 3: LZ77 lossless compression, implemented from first principles.

Idea
----
Text and data often repeat earlier material.  LZ77 (Ziv & Lempel, 1977) replaces a repeated
substring by a *back-reference* into a sliding *search window* of already-processed data.

Token convention (precise)
--------------------------
A token is a triple ``(offset, length, next_byte)``:

* ``offset``  distance BACKWARD from the current output position to the start of the match.
              For a match (length > 0): 1 <= offset <= min(window_size, position).
              For a literal (length == 0): offset is 0.
* ``length``  number of bytes copied from ``output[position - offset :]``.
* ``next_byte``  a ``bytes`` object of length 1 holding the literal byte that follows the
              match, or ``b""`` ONLY in an *end-of-input token* (a match that reaches the
              last byte of the input, so there is nothing left to append).  This optional
              byte is what avoids artificial padding at the end of the data.

A token therefore covers ``length + len(next_byte)`` input bytes, and the compressor keeps
that number <= ``lookahead_size`` (the lookahead buffer holds the bytes still to be coded).
Consequently a match followed by a literal has length <= lookahead_size - 1, while an
end-of-input match may use the whole lookahead.

Overlapping matches: when offset < length the copy source runs into bytes the same token is
producing (e.g. "aaaa" = literal 'a' + token (1, 3, b"")).  Decompression therefore copies
ONE BYTE AT A TIME, so each copied byte may itself have been produced by this very token.

Search policy (deterministic)
-----------------------------
At each position the encoder picks the LONGEST match whose start lies in the window
[position - window_size, position - 1] and, among equally long matches, the one with the
SMALLEST offset (closest to the current position).  Any match length >= 1 is allowed.
Existence of a match is monotone in its length (a match of length L contains one of length
L-1), which lets the fast engine binary-search the length.

Two engines, one specification
------------------------------
``lz77_compress_reference`` is an explicit double loop (O(window * lookahead) per position)
meant to be read.  ``lz77_compress`` (the default) uses ``bytes.rfind`` purely as a
substring-search primitive for speed; the window, lookahead, longest-match rule, tie-break
and token format are all implemented here.  The test suite requires identical tokens.

Serialized size (why token counts are not sizes)
------------------------------------------------
A token count says nothing about bytes.  We define a concrete fixed-width bit format:

    offset : offset_bits = window_size.bit_length()  bits   (values 0..window_size)
    length : length_bits = lookahead_size.bit_length() bits (values 0..lookahead_size)
    flag   : 1 bit, 1 if a next byte is present
    next   : 8 bits, only if flag == 1

preceded by a one-byte header giving the number of padding bits (0..7) appended to reach a
whole byte.  window_size and lookahead_size must be agreed out of band.  With the defaults
(4096, 18) a literal costs 13 + 5 + 1 + 8 = 27 bits for ONE input byte, so data without
repeats EXPANDS.  Compression ratio R = original size / compressed size; R < 1 means
expansion.  An empty token list serialises to ``b""``.

Complexity: reference engine O(n * W * L); fast engine O(n * log L * W) with a small constant
(the C-level search dominates).  Not a universal bound: both depend on the data.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

Token = tuple[int, int, bytes]

DEFAULT_WINDOW_SIZE = 4096
DEFAULT_LOOKAHEAD_SIZE = 18
MAX_WINDOW_SIZE = 32768
MAX_LOOKAHEAD_SIZE = 255
HEADER_BYTES = 1
DEFAULT_MAX_OUTPUT = 16 * 1024 * 1024        # decompression-bomb guard


# --------------------------------------------------------------------------- validation
def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _validate_params(window_size: int, lookahead_size: int) -> None:
    for name, value, upper in (("window_size", window_size, MAX_WINDOW_SIZE),
                               ("lookahead_size", lookahead_size, MAX_LOOKAHEAD_SIZE)):
        if not _is_int(value):
            raise TypeError(f"{name} must be an int")
        if not 1 <= value <= upper:
            raise ValueError(f"{name} must be in the range 1..{upper}, got {value}")


def _validate_data(data: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    return bytes(data)


def _check_token_shape(token: Any, index: int) -> tuple[int, int, bytes]:
    if not isinstance(token, (tuple, list)) or len(token) != 3:
        raise ValueError(f"token {index} must be a triple (offset, length, next_byte)")
    offset, length, nxt = token
    if not _is_int(offset) or not _is_int(length):
        raise ValueError(f"token {index}: offset and length must be integers")
    if not isinstance(nxt, bytes):
        raise ValueError(f"token {index}: next_byte must be bytes")
    if offset < 0 or length < 0:
        raise ValueError(f"token {index}: offset and length must be non-negative")
    if len(nxt) > 1:
        raise ValueError(f"token {index}: next_byte must be empty or exactly one byte")
    if length == 0 and offset != 0:
        raise ValueError(f"token {index}: a literal (length 0) must have offset 0")
    if length > 0 and offset == 0:
        raise ValueError(f"token {index}: a match (length > 0) needs offset >= 1")
    if length == 0 and not nxt:
        raise ValueError(f"token {index}: empty token (0, 0, b'') makes no progress")
    return offset, length, nxt


# --------------------------------------------------------------------------- match search
def _limits(i: int, n: int, window_size: int, lookahead_size: int) -> tuple[int, int]:
    """(window start index, maximum match length) for position i."""
    remaining = n - i
    max_len = remaining if remaining <= lookahead_size else lookahead_size - 1
    return max(0, i - window_size), max_len


def _longest_match_reference(data: bytes, i: int, start: int, max_len: int) -> tuple[int, int]:
    """Explicit scan of every window position; returns (length, offset), (0, 0) if none."""
    best_len, best_off = 0, 0
    for s in range(i - 1, start - 1, -1):          # ascending offset = descending start
        k = 0
        while k < max_len and data[s + k] == data[i + k]:
            k += 1
        if k > best_len:                            # strict: ties keep the smaller offset
            best_len, best_off = k, i - s
            if best_len == max_len:
                break
    return best_len, best_off


def _longest_match_fast(data: bytes, i: int, start: int, max_len: int) -> tuple[int, int]:
    """Same result as the reference scan, via binary search on the match length."""
    if max_len <= 0 or i <= start:
        return 0, 0
    lo, hi, best_s = 0, max_len, -1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        # rfind returns the HIGHEST start s with data[s:s+mid] == data[i:i+mid] and
        # start <= s <= i-1 (end bound i-1+mid forbids s >= i); overlap into the lookahead is allowed.
        s = data.rfind(data[i:i + mid], start, i - 1 + mid)
        if s != -1:
            lo, best_s = mid, s
        else:
            hi = mid - 1
    return (lo, i - best_s) if lo > 0 else (0, 0)


def _compress(data: bytes, window_size: int, lookahead_size: int,
              matcher: Callable[[bytes, int, int, int], tuple[int, int]]) -> list[Token]:
    n = len(data)
    tokens: list[Token] = []
    i = 0
    while i < n:
        start, max_len = _limits(i, n, window_size, lookahead_size)
        length, offset = matcher(data, i, start, max_len)   # contract: (0, 0) when nothing matches
        # The slice below is b"" exactly when the match reaches the end of the input: that is
        # the end-of-input token (no next byte, no artificial padding), and the loop then ends.
        tokens.append((offset, length, data[i + length:i + length + 1]))
        i += length + 1
    return tokens


def lz77_compress(data: bytes, window_size: int = DEFAULT_WINDOW_SIZE,
                  lookahead_size: int = DEFAULT_LOOKAHEAD_SIZE) -> list[Token]:
    """LZ77-compress ``data`` into (offset, length, next_byte) tokens (see module docstring)."""
    data = _validate_data(data)
    _validate_params(window_size, lookahead_size)
    return _compress(data, window_size, lookahead_size, _longest_match_fast)


def lz77_compress_reference(data: bytes, window_size: int = DEFAULT_WINDOW_SIZE,
                            lookahead_size: int = DEFAULT_LOOKAHEAD_SIZE) -> list[Token]:
    """Readable brute-force engine; must produce exactly the same tokens as ``lz77_compress``."""
    data = _validate_data(data)
    _validate_params(window_size, lookahead_size)
    return _compress(data, window_size, lookahead_size, _longest_match_reference)


# --------------------------------------------------------------------------- decompression
def lz77_decompress(tokens: Sequence[Token], *, window_size: Optional[int] = None,
                    lookahead_size: Optional[int] = None,
                    max_output: int = DEFAULT_MAX_OUTPUT) -> bytes:
    """Rebuild the original bytes from tokens.

    Validates every token (shape, ranges, offset not pointing before the start of the output,
    empty ``next_byte`` only on the final token) and refuses to produce more than ``max_output``
    bytes.  If ``window_size`` / ``lookahead_size`` are given they are enforced as well.
    """
    if not isinstance(tokens, (list, tuple)):
        raise TypeError("tokens must be a list of (offset, length, next_byte) triples")
    out = bytearray()
    last = len(tokens) - 1
    for idx, token in enumerate(tokens):
        offset, length, nxt = _check_token_shape(token, idx)
        if not nxt and idx != last:
            raise ValueError(f"token {idx}: an empty next_byte is only allowed on the final token")
        if window_size is not None and offset > window_size:
            raise ValueError(f"token {idx}: offset {offset} exceeds window_size {window_size}")
        if lookahead_size is not None and length + len(nxt) > lookahead_size:
            raise ValueError(f"token {idx}: covers {length + len(nxt)} bytes, "
                             f"more than lookahead_size {lookahead_size}")
        if offset > len(out):
            raise ValueError(f"token {idx}: offset {offset} points before the start of the output "
                             f"({len(out)} bytes produced so far)")
        if len(out) + length + len(nxt) > max_output:
            raise ValueError(f"output would exceed the limit of {max_output} bytes")
        if length:
            src = len(out) - offset
            for k in range(length):                 # byte by byte: supports overlapping matches
                out.append(out[src + k])
        out += nxt
    return bytes(out)


# --------------------------------------------------------------------------- serialisation
def _bit_widths(window_size: int, lookahead_size: int) -> tuple[int, int]:
    return window_size.bit_length(), lookahead_size.bit_length()


def token_bit_cost(token: Token, window_size: int, lookahead_size: int) -> int:
    ob, lb = _bit_widths(window_size, lookahead_size)
    return ob + lb + 1 + 8 * len(token[2])


def serialize_tokens(tokens: Sequence[Token], window_size: int = DEFAULT_WINDOW_SIZE,
                     lookahead_size: int = DEFAULT_LOOKAHEAD_SIZE) -> bytes:
    """Pack tokens with the fixed-width format described in the module docstring."""
    _validate_params(window_size, lookahead_size)
    if not isinstance(tokens, (list, tuple)):
        raise TypeError("tokens must be a list")
    if not tokens:
        return b""
    ob, lb = _bit_widths(window_size, lookahead_size)
    parts: list[str] = []
    for idx, token in enumerate(tokens):
        offset, length, nxt = _check_token_shape(token, idx)
        if offset > window_size:
            raise ValueError(f"token {idx}: offset {offset} exceeds window_size {window_size}")
        if length > lookahead_size:
            raise ValueError(f"token {idx}: length {length} exceeds lookahead_size {lookahead_size}")
        parts.append(format(offset, f"0{ob}b") + format(length, f"0{lb}b"))
        parts.append("1" + format(nxt[0], "08b") if nxt else "0")
    bits = "".join(parts)
    pad = (-len(bits)) % 8
    bits += "0" * pad
    body = int(bits, 2).to_bytes(len(bits) // 8, "big")
    return bytes([pad]) + body


def deserialize_tokens(data: bytes, window_size: int = DEFAULT_WINDOW_SIZE,
                       lookahead_size: int = DEFAULT_LOOKAHEAD_SIZE) -> list[Token]:
    """Inverse of :func:`serialize_tokens`; rejects truncated or inconsistent streams."""
    data = _validate_data(data)
    _validate_params(window_size, lookahead_size)
    if not data:
        return []
    pad = data[0]
    if pad > 7:
        raise ValueError(f"header padding count must be 0..7, got {pad}")
    body = data[1:]
    total_bits = 8 * len(body) - pad
    if total_bits <= 0:
        raise ValueError("stream contains no token bits")
    bits = format(int.from_bytes(body, "big"), f"0{8 * len(body)}b")
    if bits[total_bits:].strip("0"):
        raise ValueError("padding bits must be zero")
    bits = bits[:total_bits]
    ob, lb = _bit_widths(window_size, lookahead_size)
    tokens: list[Token] = []
    pos = 0
    while pos < total_bits:
        if pos + ob + lb + 1 > total_bits:
            raise ValueError("truncated token")
        offset = int(bits[pos:pos + ob], 2)
        length = int(bits[pos + ob:pos + ob + lb], 2)
        flag = bits[pos + ob + lb]
        pos += ob + lb + 1
        if flag == "1":
            if pos + 8 > total_bits:
                raise ValueError("truncated next byte")
            nxt = bytes([int(bits[pos:pos + 8], 2)])
            pos += 8
        else:
            nxt = b""
        if offset > window_size or length > lookahead_size:
            raise ValueError("token field exceeds the declared window/lookahead size")
        tokens.append((offset, length, nxt))
    return tokens


# --------------------------------------------------------------------------- statistics
def compression_stats(original_size: int, tokens: Sequence[Token],
                      window_size: int = DEFAULT_WINDOW_SIZE,
                      lookahead_size: int = DEFAULT_LOOKAHEAD_SIZE) -> dict[str, Any]:
    """Size statistics based on the serialised format (not on the token count).

    R = original / compressed (None when undefined); space saved = (1 - compressed/original)
    * 100 (negative means expansion; None for empty input).
    """
    _validate_params(window_size, lookahead_size)
    ob, lb = _bit_widths(window_size, lookahead_size)
    total_bits = sum(token_bit_cost(t, window_size, lookahead_size) for t in tokens)
    compressed = (HEADER_BYTES + (total_bits + 7) // 8) if tokens else 0
    literals = sum(1 for t in tokens if t[1] == 0)
    return {
        "original_bytes": original_size,
        "token_count": len(tokens),
        "literal_count": literals,
        "match_count": len(tokens) - literals,
        "offset_bits": ob,
        "length_bits": lb,
        "payload_bits": total_bits,
        "header_bytes": HEADER_BYTES if tokens else 0,
        "compressed_bytes": compressed,
        "compression_ratio": (original_size / compressed) if compressed else None,
        "space_saved_percent": ((1 - compressed / original_size) * 100) if original_size else None,
        "expanded": compressed > original_size,
    }


def describe_tokens(data: bytes, tokens: Sequence[Token], window_size: int = DEFAULT_WINDOW_SIZE,
                    lookahead_size: int = DEFAULT_LOOKAHEAD_SIZE,
                    limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Per-token description for the sliding-window illustration (JSON-compatible).

    ``position`` is the index in ``data`` where the token starts.  Raises ValueError if the
    tokens do not cover ``data`` exactly.
    """
    data = _validate_data(data)
    _validate_params(window_size, lookahead_size)
    out: list[dict[str, Any]] = []
    pos = 0
    for idx, token in enumerate(tokens):
        offset, length, nxt = _check_token_shape(token, idx)
        covers = length + len(nxt)
        if pos + covers > len(data):
            raise ValueError("tokens cover more bytes than the data contains")
        if limit is None or idx < limit:
            out.append({
                "index": idx,
                "position": pos,
                "kind": "literal" if length == 0 else ("match-end" if not nxt else "match"),
                "offset": offset,
                "length": length,
                "next_hex": nxt.hex(),
                "next_char": chr(nxt[0]) if nxt and 0x20 <= nxt[0] < 0x7F else None,
                "covers": covers,
                "match_source_start": pos - offset if length else None,
                "match_hex": data[pos:pos + length].hex(),
                "overlapping": bool(length and offset < length),
                "window_start": max(0, pos - window_size),
                "lookahead_end": min(len(data), pos + lookahead_size),
                "bits": token_bit_cost(token, window_size, lookahead_size),
            })
        pos += covers
    if pos != len(data):
        raise ValueError("tokens do not cover the data exactly")
    return out

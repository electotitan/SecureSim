"""
compression.py — LZ77 sliding-window compression (Information Theory / Combinatorics).

Mathematical background
------------------------
LZ77 exploits the fact that most real data contains repeated
substrings. It replaces a repeated substring with a back-reference
(offset, length) into a bounded "window" of already-seen data, plus
the one literal byte that follows (so the algorithm always makes
forward progress even when no match is found: length=0 then just
emits the next byte as a literal).

Formally, at each position i we search the window
    W = data[max(0, i-window_size) : i]
for the longest string that also appears as a prefix of the
lookahead buffer
    L = data[i : i+lookahead_size]
This is a longest-common-prefix search; here it is implemented with a
straightforward (but window-bounded, hence still efficient enough for
demo-sized inputs) brute-force scan for pedagogical clarity.

Each output token is a triple (offset, length, next_byte):
    offset      = how far back the match starts (0 if no match)
    length      = length of the match (0 if no match)
    next_byte   = the literal byte immediately following the match

This directly bounds the compression ratio achievable: a token costs a
fixed overhead (encoded here as an offset/length pair) regardless of
match length, so LZ77 wins whenever repeated runs are longer than that
overhead amortizes to -- the classic space/redundancy trade-off at the
heart of Shannon-style compression.
"""

from __future__ import annotations

DEFAULT_WINDOW_SIZE = 4096
DEFAULT_LOOKAHEAD_SIZE = 18  # small buffer keeps triples' "length" field compact


def lz77_compress(
    data: bytes,
    window_size: int = DEFAULT_WINDOW_SIZE,
    lookahead_size: int = DEFAULT_LOOKAHEAD_SIZE,
) -> list[tuple[int, int, int]]:
    """
    Compress `data` into a list of (offset, length, next_byte) tokens.

    offset == 0 and length == 0 means "no match found here, next_byte
    is a raw literal". Otherwise, the decoder should copy `length`
    bytes starting `offset` bytes before the current output position,
    then append `next_byte` (which may be omitted at end-of-input --
    handled below by allowing next_byte to be None only in that case).
    """
    tokens: list[tuple[int, int, int]] = []
    i = 0
    n = len(data)
    while i < n:
        best_offset = 0
        best_length = 0
        window_start = max(0, i - window_size)

        # Cap the match length at n-i-1 (not n-i): this deliberately reserves
        # at least one byte so a literal next_byte is always available after
        # every match, which keeps every token's next_byte a valid 0-255
        # value (no end-of-input sentinel needed) -- important because
        # tokens are serialized to raw bytes for the encryption stage below.
        max_len = min(lookahead_size, n - i - 1)

        # Brute-force search: try every possible match start in the window.
        for j in range(window_start, i):
            length = 0
            # LZ77 allows overlapping matches (the reference can point into
            # bytes that are themselves being copied), hence data[j+length]
            # is compared against data[i+length], not a frozen snapshot.
            while length < max_len and data[j + length] == data[i + length]:
                length += 1
            if length > best_length:
                best_length = length
                best_offset = i - j

        if best_length > 0:
            next_byte = data[i + best_length]
            tokens.append((best_offset, best_length, next_byte))
            i += best_length + 1
        else:
            tokens.append((0, 0, data[i]))
            i += 1

    return tokens


def lz77_decompress(tokens: list[tuple[int, int, int]]) -> bytes:
    """
    Reconstruct the original bytes from LZ77 tokens by literally
    replaying each (offset, length, next_byte) instruction: copy
    `length` bytes from `offset` bytes back in the *output so far*
    (not the input -- the decoder never sees the original data), then
    append next_byte (every token carries a valid literal byte -- see
    lz77_compress's max_len bound).
    """
    output = bytearray()
    for offset, length, next_byte in tokens:
        if length > 0:
            start = len(output) - offset
            for k in range(length):
                # byte-by-byte copy (not a slice) because matches may overlap
                # the region being written, e.g. compressing "aaaaaa".
                output.append(output[start + k])
        output.append(next_byte)
    return bytes(output)


def serialize_tokens(tokens: list[tuple[int, int, int]]) -> bytes:
    """
    Pack each (offset, length, next_byte) token into 4 raw bytes
    (offset as 2 bytes big-endian, then length, then next_byte), so the
    compressed representation can be handed to the encryption stage as
    an ordinary byte string. This is the conventional final step of a
    real LZ77-based format (e.g. DEFLATE): the token stream itself is
    just another byte sequence once its fields are fixed-width.
    """
    out = bytearray()
    for offset, length, next_byte in tokens:
        if not (0 <= offset <= 0xFFFF):
            raise ValueError(f"offset {offset} does not fit in 2 bytes; reduce window_size")
        if not (0 <= length <= 0xFF):
            raise ValueError(f"length {length} does not fit in 1 byte; reduce lookahead_size")
        out.append((offset >> 8) & 0xFF)
        out.append(offset & 0xFF)
        out.append(length & 0xFF)
        out.append(next_byte & 0xFF)
    return bytes(out)


def deserialize_tokens(data: bytes) -> list[tuple[int, int, int]]:
    """Inverse of serialize_tokens: unpack 4-byte groups back into
    (offset, length, next_byte) tuples."""
    if len(data) % 4 != 0:
        raise ValueError("serialized token data length must be a multiple of 4")
    tokens = []
    for i in range(0, len(data), 4):
        offset = (data[i] << 8) | data[i + 1]
        length = data[i + 2]
        next_byte = data[i + 3]
        tokens.append((offset, length, next_byte))
    return tokens


def compression_report(data: bytes, window_size: int = DEFAULT_WINDOW_SIZE,
                        lookahead_size: int = DEFAULT_LOOKAHEAD_SIZE) -> dict:
    """Convenience wrapper producing the API response payload for this stage."""
    tokens = lz77_compress(data, window_size, lookahead_size)
    original_size = len(data)
    # Each token is modeled as: offset (2 bytes) + length (1 byte) + next_byte (1 byte) = 4 bytes
    compressed_size = len(tokens) * 4
    ratio = (original_size / compressed_size) if compressed_size > 0 else float("inf")
    return {
        "tokens": [{"offset": o, "length": l, "next_byte": b} for o, l, b in tokens],
        "num_tokens": len(tokens),
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": round(ratio, 4) if compressed_size > 0 else None,
    }

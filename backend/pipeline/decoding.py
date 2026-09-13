"""
decoding.py — Final receiver-side decoding stage.

Runs the source-side transformations in reverse:
    LZ77 tokens -> bytes -> binary string -> Huffman-decode -> original text
and performs the definitive end-to-end correctness check: byte-for-byte
comparison of the recovered plaintext against the original input. This
is the pipeline's ground truth for whether "the message got through" --
every earlier stage (channel noise, error correction) only ever
*estimates* success via syndromes/checksums; this stage is where that
estimate is confirmed (or refuted) against the real original message,
which in a real deployed system the receiver would never have access
to, but which this educational simulator keeps around specifically to
demonstrate the difference between "error correction reported success"
and "the message was actually, provably recovered correctly".
"""

from __future__ import annotations
from .compression import lz77_decompress
from .source_encoding import huffman_decode
from .binary_repr import binary_to_text


def decode_full(
    lz77_tokens: list[tuple[int, int, int]],
    huffman_code_table: dict[str, str],
    original_text: str,
) -> dict:
    """
    Full reverse pipeline: LZ77 decompress -> bytes -> bitstring ->
    Huffman decode -> recovered text, then compare byte-for-byte
    against `original_text`.
    """
    decompressed_bytes = lz77_decompress(lz77_tokens)
    # The Huffman-encoded bitstring was itself carried as compressed bytes;
    # reconstruct the bitstring from those bytes (inverse of packing bits
    # into bytes for transport).
    bitstring = "".join(format(b, "08b") for b in decompressed_bytes)
    # Trailing zero-padding may have been added to byte-align the bitstring;
    # huffman_decode will raise on leftover bits if padding remains
    # ambiguous, so we try progressively shorter suffixes only as a
    # last resort compatibility measure -- callers that track exact
    # bit-length should prefer passing an exact bitstring instead.
    try:
        recovered_text = huffman_decode(bitstring, huffman_code_table)
    except ValueError:
        recovered_text = _decode_with_padding_trim(bitstring, huffman_code_table)

    success = recovered_text == original_text
    diff_positions = []
    if not success:
        max_len = max(len(recovered_text), len(original_text))
        for i in range(max_len):
            orig_char = original_text[i] if i < len(original_text) else None
            rec_char = recovered_text[i] if i < len(recovered_text) else None
            if orig_char != rec_char:
                diff_positions.append(i)

    return {
        "success": success,
        "original": original_text,
        "recovered": recovered_text,
        "diff_positions": diff_positions,
    }


def _decode_with_padding_trim(bitstring: str, code_table: dict[str, str]) -> str:
    """Try trimming up to 7 trailing padding bits until huffman_decode succeeds
    (handles byte-alignment padding added when packing the Huffman bitstring
    into whole bytes for the compression/transport stages)."""
    for trim in range(8):
        candidate = bitstring[: len(bitstring) - trim] if trim > 0 else bitstring
        try:
            return huffman_decode(candidate, code_table)
        except ValueError:
            continue
    raise ValueError("could not decode bitstring even after trimming padding")


def decode_binary_only(bits: str) -> str:
    """Simple wrapper for reversing just the binary-representation stage."""
    return binary_to_text(bits)

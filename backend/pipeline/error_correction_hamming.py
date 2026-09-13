"""
error_correction_hamming.py — Hamming(7,4) code (Coding Theory).

Mathematical background
------------------------
Hamming(7,4) encodes 4 data bits (d1 d2 d3 d4) into a 7-bit codeword
by adding 3 parity bits (p1 p2 p3), positioned so that each parity bit
covers a distinct, overlapping subset of the data bits (bit positions
whose binary index has a particular bit set). This is a linear block
code over GF(2): every codeword is a linear combination (via a
generator matrix G) of the data bits, and the code has minimum Hamming
distance 3 between any two distinct codewords -- which by the standard
coding-theory bound (distance >= 2t+1 to correct t errors) is exactly
enough to *correct any single-bit error* (t=1) and *detect* (but not
correct) any double-bit error.

We use the standard non-systematic bit layout (positions 1-indexed):
    position: 1  2  3  4  5  6  7
    content:  p1 p2 d1 p3 d2 d3 d4
Parity bit p_k covers exactly the positions whose binary
representation has bit (k-1) set (p1 covers positions with bit0 set:
1,3,5,7; p2 covers bit1: 2,3,6,7; p3 covers bit2: 4,5,6,7) -- this is
what makes the *syndrome* (the pattern of parity-check failures at the
receiver) directly equal, in binary, to the 1-indexed position of any
single flipped bit -- the key trick that makes decoding a simple
lookup rather than a search.

Generator matrix G (4x7, rows = how each data bit contributes to each
codeword position) and parity-check matrix H (3x7, rows = which
codeword positions each parity bit checks) are built explicitly below
to make this algebraic structure visible, though the encode/decode
functions also work directly off the bit-position definitions above
for clarity.
"""

from __future__ import annotations

# Generator matrix G: codeword = data @ G (mod 2), data = [d1,d2,d3,d4],
# codeword columns ordered [p1,p2,d1,p3,d2,d3,d4] to match the layout above.
GENERATOR_MATRIX = [
    # p1 p2 d1 p3 d2 d3 d4
    [1, 1, 1, 0, 0, 0, 0],  # d1's contribution
    [1, 0, 0, 1, 1, 0, 0],  # d2's contribution
    [0, 1, 0, 1, 0, 1, 0],  # d3's contribution
    [1, 1, 0, 1, 0, 0, 1],  # d4's contribution
]

# Parity-check matrix H (3x7): H @ codeword^T = syndrome (mod 2).
# Column j of H is the binary representation of position (j+1).
PARITY_CHECK_MATRIX = [
    [1, 0, 1, 0, 1, 0, 1],  # checks bit0 of position -> p1
    [0, 1, 1, 0, 0, 1, 1],  # checks bit1 of position -> p2
    [0, 0, 0, 1, 1, 1, 1],  # checks bit2 of position -> p3
]


def hamming_encode(data_bits_4: str) -> str:
    """
    Encode 4 data bits into a 7-bit Hamming codeword.

    Directly applies the covering rule: p1 = d1 xor d2 xor d4,
    p2 = d1 xor d3 xor d4, p3 = d2 xor d3 xor d4 (derivable from which
    positions 3,5,6,7 -- i.e. d1,d2,d3,d4's positions -- each parity
    bit covers), matching GENERATOR_MATRIX above.
    """
    if len(data_bits_4) != 4 or any(b not in "01" for b in data_bits_4):
        raise ValueError("hamming_encode expects exactly 4 bits ('0'/'1' characters)")

    d1, d2, d3, d4 = (int(b) for b in data_bits_4)
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4

    codeword = [p1, p2, d1, p3, d2, d3, d4]  # positions 1..7
    return "".join(str(b) for b in codeword)


def hamming_decode(received_7bit: str) -> tuple[str, bool]:
    """
    Decode a (possibly corrupted) 7-bit codeword.

    Compute the syndrome via the parity-check equations; a nonzero
    syndrome, interpreted as a binary number (s3 s2 s1), directly gives
    the 1-indexed position of the single flipped bit (this is the
    defining property of the Hamming(7,4) bit layout). Flip that bit
    to correct it, then extract the 4 data bits from their known
    positions (3, 5, 6, 7).

    Returns (corrected_4_data_bits, error_was_detected_and_corrected).
    A syndrome of 0 means no error was detected (error_corrected=False
    but this also covers the "no error at all" case -- see
    hamming_decode_detailed for a 3-way distinction).
    """
    if len(received_7bit) != 7 or any(b not in "01" for b in received_7bit):
        raise ValueError("hamming_decode expects exactly 7 bits ('0'/'1' characters)")

    bits = [int(b) for b in received_7bit]  # 0-indexed list, bits[0]=position1, ..., bits[6]=position7
    p1, p2, d1, p3, d2, d3, d4 = bits

    s1 = p1 ^ d1 ^ d2 ^ d4  # re-check p1's coverage (positions 1,3,5,7)
    s2 = p2 ^ d1 ^ d3 ^ d4  # re-check p2's coverage (positions 2,3,6,7)
    s3 = p3 ^ d2 ^ d3 ^ d4  # re-check p3's coverage (positions 4,5,6,7)

    syndrome = (s3 << 2) | (s2 << 1) | s1  # binary syndrome = 1-indexed error position

    error_detected = syndrome != 0
    if error_detected:
        error_index = syndrome - 1  # convert to 0-indexed position in `bits`
        bits[error_index] ^= 1  # flip the identified bit to correct it
        p1, p2, d1, p3, d2, d3, d4 = bits

    corrected_data = f"{d1}{d2}{d3}{d4}"
    return corrected_data, error_detected


def hamming_decode_detailed(received_7bit: str) -> dict:
    """Richer decode result for the API/frontend: distinguishes no-error vs corrected."""
    bits = [int(b) for b in received_7bit]
    p1, p2, d1, p3, d2, d3, d4 = bits
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = (s3 << 2) | (s2 << 1) | s1

    corrected_bits = bits[:]
    error_position = None
    if syndrome != 0:
        error_position = syndrome  # 1-indexed
        corrected_bits[syndrome - 1] ^= 1

    p1c, p2c, d1c, p3c, d2c, d3c, d4c = corrected_bits
    return {
        "syndrome": syndrome,
        "error_position": error_position,  # None if no error, else 1-indexed position
        "corrected_codeword": "".join(str(b) for b in corrected_bits),
        "corrected_data_bits": f"{d1c}{d2c}{d3c}{d4c}",
    }


def hamming_encode_message(data_bits: str) -> str:
    """Apply hamming_encode block-wise across an arbitrary-length bitstring,
    padding the final block with zero bits if it's not a multiple of 4."""
    padded = data_bits + "0" * ((-len(data_bits)) % 4)
    blocks = [padded[i:i + 4] for i in range(0, len(padded), 4)]
    return "".join(hamming_encode(block) for block in blocks)


def hamming_decode_message(received_bits: str, original_length: int | None = None) -> tuple[str, list[bool]]:
    """
    Apply hamming_decode block-wise across an arbitrary-length received
    bitstring (must be a multiple of 7). Returns (decoded_data_bits,
    per_block_error_flags). If `original_length` is given, the result
    is truncated back to that length (undoing hamming_encode_message's
    zero-padding of the final block).
    """
    if len(received_bits) % 7 != 0:
        raise ValueError("received bitstring length must be a multiple of 7")
    blocks = [received_bits[i:i + 7] for i in range(0, len(received_bits), 7)]
    decoded_parts = []
    error_flags = []
    for block in blocks:
        data, had_error = hamming_decode(block)
        decoded_parts.append(data)
        error_flags.append(had_error)
    decoded = "".join(decoded_parts)
    if original_length is not None:
        decoded = decoded[:original_length]
    return decoded, error_flags

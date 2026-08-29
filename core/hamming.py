"""
core/hamming.py

Hamming(7,4) encoding and syndrome-based correction, built explicitly from
the parity-bit-position construction (no coding-theory library calls).

IMPORTANT ARCHITECTURAL NOTE vs. the original Phase-0 stage list:
Error correction only means something if it happens BEFORE the noisy
channel (to add redundancy) and AFTER it (to use that redundancy to
recover). So in the actual pipeline, HammingEncodeStage runs right after
packetization/CRC-attach and before routing+the channel, while
HammingCorrectStage runs right after the channel (before CRC-check /
reassembly). This module still logically corresponds to the single
"Hamming Error Correction" stage named in the abstract -- it's just split
into its two necessary halves.

Convention (1-indexed codeword positions, standard Hamming(7,4)):
    position: 1  2  3  4  5  6  7
    bit:      p1 p2 d1 p3 d2 d3 d4
    p1 = d1 ^ d2 ^ d4   (covers positions 1,3,5,7)
    p2 = d1 ^ d3 ^ d4   (covers positions 2,3,6,7)
    p3 = d2 ^ d3 ^ d4   (covers positions 4,5,6,7)

On decode, recomputing c1,c2,c3 the same way and combining them
(c1 + 2*c2 + 4*c3) gives either 0 (no error) or the 1-indexed position of
the single flipped bit -- a classic and elegant property of this code.

Forward stage (encode) : {"packets", "meta"} -> {"packets", "meta"}
    each packet's payload_bits (padded to a multiple of 4) is expanded to
    7 bits per nibble; original bit length is remembered in
    meta["cipher_bit_length"] so it can be trimmed back after correction.
Inverse stage (correct): {"packets", "meta"} -> {"packets", "meta"}
    each packet's payload_bits is corrected and truncated back to
    meta["cipher_bit_length"].

Math concept demonstrated: linear algebra over GF(2), coding theory.
Limitation this module deliberately demonstrates rather than hides:
Hamming(7,4) can only correct a SINGLE bit error per 7-bit block. Two
errors in the same block will be "corrected" to the WRONG codeword
(a real, well-known limitation of this code) -- see HammingCorrectStage.
"""

from core.pipeline import Stage, Trace


def hamming_encode_nibble(d1: int, d2: int, d3: int, d4: int) -> list[int]:
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]


def hamming_decode_codeword(bits7: list[int]) -> tuple[list[int], int]:
    """
    Returns (corrected_data_bits, syndrome).
    syndrome == 0 means no error detected.
    syndrome in 1..7 means that 1-indexed position was flipped and has
    been corrected (or, if 2 bits were actually wrong, MIScorrected --
    that's an inherent limit of a single-error-correcting code).
    """
    p1, p2, d1, p3, d2, d3, d4 = bits7
    c1 = p1 ^ d1 ^ d2 ^ d4
    c2 = p2 ^ d1 ^ d3 ^ d4
    c3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = c1 * 1 + c2 * 2 + c3 * 4

    corrected = bits7[:]
    if syndrome != 0:
        corrected[syndrome - 1] ^= 1

    p1c, p2c, d1c, p3c, d2c, d3c, d4c = corrected
    return [d1c, d2c, d3c, d4c], syndrome


class HammingEncodeStage(Stage):
    name = "Hamming Error Correction (encode)"

    def run(self, data: dict, trace: Trace) -> dict:
        new_packets = []
        for pkt in data["packets"]:
            bits = pkt["payload_bits"]
            pad = (-len(bits)) % 4
            padded = bits + "0" * pad

            encoded = []
            for i in range(0, len(padded), 4):
                nibble = [int(b) for b in padded[i:i + 4]]
                codeword = hamming_encode_nibble(*nibble)
                encoded.extend(codeword)

            encoded_bits = "".join(str(b) for b in encoded)
            new_packets.append({**pkt, "payload_bits": encoded_bits})
            trace.log_step(
                action="encode_packet", seq=pkt["seq"],
                data_bits=len(bits), codeword_bits=len(encoded_bits),
            )

        return {"packets": new_packets, "meta": data["meta"]}


class HammingCorrectStage(Stage):
    name = "Hamming Error Correction (correct)"

    def run(self, data: dict, trace: Trace) -> dict:
        original_len = data["meta"]["cipher_bit_length"]
        new_packets = []
        for pkt in data["packets"]:
            bits = pkt["payload_bits"]
            recovered_bits = []
            for block_idx, i in enumerate(range(0, len(bits), 7)):
                codeword = [int(b) for b in bits[i:i + 7]]
                data_nibble, syndrome = hamming_decode_codeword(codeword)
                recovered_bits.extend(data_nibble)
                if syndrome != 0:
                    trace.log_step(
                        action="correct_bit", seq=pkt["seq"], block=block_idx,
                        syndrome=syndrome, corrected_position=syndrome,
                    )
                else:
                    trace.log_step(action="no_error", seq=pkt["seq"], block=block_idx)

            recovered_str = "".join(str(b) for b in recovered_bits)[:original_len]
            new_packets.append({**pkt, "payload_bits": recovered_str})

        return {"packets": new_packets, "meta": data["meta"]}

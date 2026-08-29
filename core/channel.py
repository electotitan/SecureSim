"""
core/channel.py

Binary Symmetric Channel: every bit of every packet's payload is flipped
independently with probability p, using Python's random module directly
(no numpy black box) so the per-bit Bernoulli trial is explicit.

This stage is where corruption actually enters the pipeline -- everything
downstream (CRC-check, Hamming-correct) exists to detect/undo what this
stage does.

Math concept demonstrated: probability theory (Bernoulli trials, expected
bit-error rate).
"""

import random

from core.pipeline import Stage, Trace


class ChannelStage(Stage):
    name = "Channel Transmission (Noise)"

    def __init__(self, flip_probability: float = 0.0):
        self.p = flip_probability

    def run(self, data: dict, trace: Trace) -> dict:
        new_packets = []
        total_bits = 0
        total_flips = 0

        for pkt in data["packets"]:
            bits = list(pkt["payload_bits"])
            for i in range(len(bits)):
                total_bits += 1
                if random.random() < self.p:
                    bits[i] = "1" if bits[i] == "0" else "0"
                    total_flips += 1
                    trace.log_step(action="bit_flip", seq=pkt["seq"], bit_index=i)
            new_packets.append({**pkt, "payload_bits": "".join(bits)})

        trace.metadata["configured_p"] = self.p
        trace.metadata["total_bits"] = total_bits
        trace.metadata["total_flips"] = total_flips
        trace.metadata["empirical_flip_rate"] = (total_flips / total_bits) if total_bits else 0.0

        return {"packets": new_packets, "meta": data["meta"]}

"""
core/packetization.py

Splits RSA ciphertext blocks into individually addressed packets (each
carrying one ciphertext block as a fixed-width bit string plus a sequence
number), and reassembles them on the receive side after routing/channel/
error-correction.

Shared metadata that ISN'T per-packet (the code table, block widths, etc.)
travels alongside the packet list rather than being duplicated into every
packet -- in a real network this would be a session/handshake concern, but
keeping it explicit here makes the data flow easy to trace.

Forward stage : {"cipher_blocks", "block_bits", "last_block_len",
                 "code_table", "num_symbols", "cipher_bit_length"}
             -> {"packets": [...], "meta": {...}}
Inverse stage : {"packets": [...], "meta": {...}}
             -> {"cipher_blocks", "block_bits", "last_block_len",
                 "code_table", "num_symbols"}

Math concept demonstrated: data structuring / sequencing.
"""

from core.pipeline import Stage, Trace


class PacketizationStage(Stage):
    name = "Packetization"

    def run(self, data: dict, trace: Trace) -> dict:
        cipher_bit_length = data["cipher_bit_length"]
        packets = []
        for i, c in enumerate(data["cipher_blocks"]):
            payload_bits = format(c, f"0{cipher_bit_length}b")
            packets.append({"seq": i, "payload_bits": payload_bits})
            trace.log_step(action="create_packet", seq=i, payload_bits=payload_bits)

        meta = {
            "block_bits": data["block_bits"],
            "last_block_len": data["last_block_len"],
            "code_table": data["code_table"],
            "num_symbols": data["num_symbols"],
            "cipher_bit_length": cipher_bit_length,
        }
        trace.metadata["num_packets"] = len(packets)
        return {"packets": packets, "meta": meta}


class ReassemblyStage(Stage):
    name = "Packet Reassembly"

    def run(self, data: dict, trace: Trace) -> dict:
        packets = sorted(data["packets"], key=lambda p: p["seq"])
        meta = data["meta"]

        cipher_blocks = []
        for p in packets:
            value = int(p["payload_bits"], 2) if p["payload_bits"] else 0
            cipher_blocks.append(value)
            trace.log_step(action="reassemble_packet", seq=p["seq"], value=value)

        return {
            "cipher_blocks": cipher_blocks,
            "block_bits": meta["block_bits"],
            "last_block_len": meta["last_block_len"],
            "code_table": meta["code_table"],
            "num_symbols": meta["num_symbols"],
        }

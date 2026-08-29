"""
core/crc.py

CRC-8 error detection via polynomial division over GF(2), implemented as
manual XOR-based binary long division (no crc/binascii library calls).

Generator polynomial: x^8 + x^2 + x + 1  ->  bit pattern 1_0000_0111
(9 bits, since it represents a degree-8 polynomial).

Attach stage : append an 8-bit CRC remainder to each packet, computed over
               that packet's (Hamming-encoded) payload bits.
Check stage  : recompute the remainder over (payload_bits + crc); a result
               of all zeros means "no error detected" (not "no error" --
               CRC can miss certain error patterns, which is worth noting
               in the report).

Math concept demonstrated: finite field (GF(2)) polynomial arithmetic.
"""

from core.pipeline import Stage, Trace

CRC8_POLY = "100000111"  # x^8 + x^2 + x + 1


def _xor_divide(bits: str, poly: str, trace: Trace | None = None, log_prefix: str = "") -> str:
    """
    Binary polynomial long division via repeated XOR.
    Returns the remainder (len(poly) - 1 bits).
    """
    data = list(bits)
    poly_bits = [int(b) for b in poly]
    n = len(poly_bits)

    for i in range(len(bits) - n + 1):
        if data[i] == "1":
            for j in range(n):
                data[i + j] = str(int(data[i + j]) ^ poly_bits[j])
            if trace is not None:
                trace.log_step(action="xor_step", context=log_prefix, position=i)

    return "".join(data[-(n - 1):])


def compute_crc(payload_bits: str, trace: Trace | None = None, log_prefix: str = "") -> str:
    """Generate the CRC remainder to attach: divide (message + 8 zero bits)."""
    return _xor_divide(payload_bits + "0" * 8, CRC8_POLY, trace, log_prefix)


def verify_crc(payload_bits: str, crc: str, trace: Trace | None = None, log_prefix: str = "") -> bool:
    """Verify: divide (message + received crc); remainder of all zeros = OK."""
    remainder = _xor_divide(payload_bits + crc, CRC8_POLY, trace, log_prefix)
    return int(remainder, 2) == 0


class CRCAttachStage(Stage):
    name = "CRC Error Detection (attach)"

    def run(self, data: dict, trace: Trace) -> dict:
        new_packets = []
        for pkt in data["packets"]:
            crc = compute_crc(pkt["payload_bits"], trace, log_prefix=f"seq{pkt['seq']}-attach")
            new_packets.append({**pkt, "crc": crc})
            trace.log_step(action="attach_crc", seq=pkt["seq"], crc=crc)
        return {"packets": new_packets, "meta": data["meta"]}


class CRCCheckStage(Stage):
    name = "CRC Error Detection (check)"

    def run(self, data: dict, trace: Trace) -> dict:
        new_packets = []
        flagged = 0
        for pkt in data["packets"]:
            ok = verify_crc(pkt["payload_bits"], pkt["crc"], trace, log_prefix=f"seq{pkt['seq']}-check")
            if not ok:
                flagged += 1
            new_packets.append({**pkt, "crc_ok": ok})
            trace.log_step(action="check_crc", seq=pkt["seq"], crc_ok=ok)
        trace.metadata["packets_flagged_by_crc"] = flagged
        return {"packets": new_packets, "meta": data["meta"]}

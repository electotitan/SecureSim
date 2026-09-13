"""
packetization.py — Split a byte stream into fixed-size packets and
reassemble them.

Real networks (IP/TCP) cannot send arbitrarily large messages as one
unit -- they carve data into packets bounded by the network's Maximum
Transmission Unit. Each packet here carries a small header
(sequence number, total packet count, a simple header checksum) so the
receiver can detect reordering, loss, and (for the checksum) simple
corruption -- independent of, and prior to, the dedicated CRC-32
error-detection stage that follows in the pipeline.
"""

from __future__ import annotations


def _header_checksum(seq_num: int, total_packets: int, payload: bytes) -> int:
    """
    A simple additive checksum over the header fields and payload
    bytes, mod 256 -- intentionally lightweight (this is a per-packet
    sanity check; end-to-end integrity is CRC-32's job in the next
    pipeline stage). checksum = (seq_num + total_packets + sum(payload)) mod 256.
    """
    total = seq_num + total_packets + sum(payload)
    return total % 256


def create_packets(data: bytes, payload_size: int = 64) -> list[dict]:
    """
    Split `data` into packets of at most `payload_size` bytes each.
    Packet = {seq_num, total_packets, payload, header_checksum}.
    """
    if payload_size <= 0:
        raise ValueError("payload_size must be positive")

    if len(data) == 0:
        total_packets = 1
        chunks = [b""]
    else:
        chunks = [data[i:i + payload_size] for i in range(0, len(data), payload_size)]
        total_packets = len(chunks)

    packets = []
    for seq_num, payload in enumerate(chunks):
        packets.append({
            "seq_num": seq_num,
            "total_packets": total_packets,
            "payload": payload,
            "header_checksum": _header_checksum(seq_num, total_packets, payload),
        })
    return packets


def verify_packet(packet: dict) -> bool:
    """Recompute the header checksum and compare -- detects header/payload corruption."""
    expected = _header_checksum(packet["seq_num"], packet["total_packets"], packet["payload"])
    return expected == packet["header_checksum"]


def reassemble_packets(packets: list[dict]) -> bytes:
    """Sort packets by sequence number and concatenate their payloads."""
    if not packets:
        return b""
    ordered = sorted(packets, key=lambda p: p["seq_num"])
    total_packets = ordered[0]["total_packets"]
    seq_nums = [p["seq_num"] for p in ordered]
    if seq_nums != list(range(total_packets)):
        missing = sorted(set(range(total_packets)) - set(seq_nums))
        raise ValueError(f"cannot reassemble: missing packet sequence numbers {missing}")
    return b"".join(p["payload"] for p in ordered)


def packetize_report(data: bytes, payload_size: int = 64) -> dict:
    packets = create_packets(data, payload_size)
    return {
        "packets": [
            {
                "seq_num": p["seq_num"],
                "total_packets": p["total_packets"],
                "payload_hex": p["payload"].hex(),
                "payload_size": len(p["payload"]),
                "header_checksum": p["header_checksum"],
            }
            for p in packets
        ],
        "num_packets": len(packets),
    }

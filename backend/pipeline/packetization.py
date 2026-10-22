"""Stage 6: packetization (framing, sequence numbers, header checksum, reassembly).

A byte string is split into fixed-size chunks; each chunk travels in a packet that also carries the
information the receiver needs to put the message back together.

Wire format (big-endian, ``HEADER_SIZE`` = 14 bytes, then the payload)
----------------------------------------------------------------------
    offset  size  field
    0       1     version          always 1
    1       1     flags            reserved, must be 0
    2       4     sequence         0-based index of this packet (0 .. total-1)
    6       4     total            number of packets in the message (>= 1)
    10      2     payload_length   number of payload bytes that follow (0 .. 65535)
    12      2     header_checksum  16-bit Internet checksum of the 14 header bytes (see below)
    14      n     payload

``serialized_size = HEADER_SIZE + len(payload)`` is NOT the payload size: the difference is protocol
overhead.  ``payload_size`` (an argument of ``create_packets``) is the maximum payload per packet.

Conventions
-----------
* Sequence numbers start at 0.  ``create_packets`` emits ceil(len(data) / payload_size) packets, all full
  except possibly the last (never an empty trailing packet).
* An EMPTY message is ONE packet with an empty payload and total = 1 (not zero packets), so a receiver can
  tell "the message is empty" from "nothing arrived".
* Reassembly accepts packets in ANY order and returns the payloads in sequence order.

Header checksum: the Internet checksum (RFC 1071)
-------------------------------------------------
Treat the data as 16-bit big-endian words (a zero byte is appended if the length is odd), add them with
END-AROUND CARRY (one's-complement addition: any carry out of bit 16 is added back in) and take the one's
complement of the result.  The checksum field itself is 0 while it is computed; a receiver sums ALL header
words including the stored checksum and must obtain 0xFFFF, i.e. the checksum of the whole header is 0.

What it detects: every single-bit error in the header (verified exhaustively over all 112 header bits in the
tests).  What it does NOT detect, by construction:
* Reordering of 16-bit words: addition is commutative, so swapping two words leaves the checksum unchanged.
* Two errors that cancel (one word increased by k, another decreased by k).
* Anything outside the header: this checksum does NOT cover the payload.  Payload integrity is the job of the
  CRC-32 stage that follows (a CRC detects reordering and burst errors that this sum cannot).

Reassembly checks, in this priority order (a corrupt header makes every other field untrustworthy)
-------------------------------------------------------------------------------------------------
1. no packets;  2. header checksum failures;  3. inconsistent ``total`` values;  4. sequence outside 0..total-1;
5. duplicates (identical retransmissions can be tolerated with ``allow_identical_duplicates``; conflicting
copies never are);  6. missing sequence numbers.  Lists in error details are bounded (a forged header may claim
total = 4,294,967,295) while exact counts are always reported.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

VERSION = 1
HEADER_SIZE = 14
DEFAULT_PAYLOAD_SIZE = 64
MAX_PAYLOAD_SIZE = 0xFFFF
MAX_SEQUENCE = 0xFFFFFFFF
DETAIL_LIST_LIMIT = 1000


# --------------------------------------------------------------------------- errors
class PacketError(ValueError):
    """Base class for packet-level failures. ``kind`` names the failure; ``details`` is JSON-compatible."""
    kind = "packet_error"

    def __init__(self, message: str, **details: Any):
        super().__init__(message)
        self.details = details


class NoPacketsError(PacketError):
    kind = "no_packets"


class HeaderChecksumError(PacketError):
    kind = "header_checksum"


class InconsistentTotalError(PacketError):
    kind = "inconsistent_total"


class InvalidSequenceError(PacketError):
    kind = "invalid_sequence"


class DuplicatePacketsError(PacketError):
    kind = "duplicate"


class MissingPacketsError(PacketError):
    kind = "missing"


class PacketFormatError(PacketError):
    """Bytes that cannot be parsed as a packet at all."""
    kind = "format"


# --------------------------------------------------------------------------- validation helpers
def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _as_bytes(value: Any, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes")
    return bytes(value)


# --------------------------------------------------------------------------- Internet checksum
def internet_checksum(data: bytes) -> int:
    """16-bit Internet checksum (RFC 1071): complement of the end-around-carry sum of 16-bit words."""
    data = _as_bytes(data, "data")
    if len(data) % 2:
        data += b"\x00"
    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) | data[i + 1]
    while total >> 16:                                   # end-around carry
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


# --------------------------------------------------------------------------- packet
def _header_bytes(sequence: int, total: int, payload_length: int, checksum: int) -> bytes:
    return (bytes([VERSION, 0]) + sequence.to_bytes(4, "big") + total.to_bytes(4, "big")
            + payload_length.to_bytes(2, "big") + checksum.to_bytes(2, "big"))


@dataclass(frozen=True)
class Packet:
    """One packet. Construction validates field RANGES only; use :meth:`header_ok` for integrity."""
    sequence: int
    total: int
    payload: bytes
    checksum: int

    def __post_init__(self) -> None:
        for name in ("sequence", "total", "checksum"):
            if not _is_int(getattr(self, name)):
                raise TypeError(f"{name} must be an int")
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        if not 0 <= self.sequence <= MAX_SEQUENCE:
            raise ValueError(f"sequence must be in 0..{MAX_SEQUENCE}")
        if not 1 <= self.total <= MAX_SEQUENCE:
            raise ValueError(f"total must be in 1..{MAX_SEQUENCE}")
        if len(self.payload) > MAX_PAYLOAD_SIZE:
            raise ValueError(f"payload must be at most {MAX_PAYLOAD_SIZE} bytes")
        if not 0 <= self.checksum <= 0xFFFF:
            raise ValueError("checksum must be in 0..65535")

    @classmethod
    def build(cls, sequence: int, total: int, payload: bytes) -> "Packet":
        """A packet with a correct header checksum."""
        payload = _as_bytes(payload, "payload")
        if not _is_int(sequence) or not _is_int(total):
            raise TypeError("sequence and total must be ints")
        if len(payload) > MAX_PAYLOAD_SIZE:
            raise ValueError(f"payload must be at most {MAX_PAYLOAD_SIZE} bytes")
        if not 0 <= sequence <= MAX_SEQUENCE or not 1 <= total <= MAX_SEQUENCE:
            raise ValueError("sequence or total out of range")
        checksum = internet_checksum(_header_bytes(sequence, total, len(payload), 0))
        return cls(sequence, total, payload, checksum)

    @property
    def payload_length(self) -> int:
        return len(self.payload)

    @property
    def serialized_size(self) -> int:
        """Header plus payload bytes (what travels on the wire)."""
        return HEADER_SIZE + len(self.payload)

    def header_bytes(self) -> bytes:
        return _header_bytes(self.sequence, self.total, len(self.payload), self.checksum)

    def header_ok(self) -> bool:
        """True iff the Internet checksum over the complete header (including the stored checksum) is 0."""
        return internet_checksum(self.header_bytes()) == 0

    def to_bytes(self) -> bytes:
        return self.header_bytes() + self.payload

    def header_fields(self) -> dict[str, int]:
        return {"version": VERSION, "flags": 0, "sequence": self.sequence, "total": self.total,
                "payload_length": len(self.payload), "checksum": self.checksum}


def packet_from_bytes(data: bytes) -> Packet:
    """Parse wire bytes. Raises ``PacketFormatError`` for structurally invalid input.

    The header checksum is NOT verified here (use ``Packet.header_ok``): a packet with a bad checksum
    must still be inspectable.
    """
    data = _as_bytes(data, "data")
    if len(data) < HEADER_SIZE:
        raise PacketFormatError(f"packet is {len(data)} bytes, shorter than the {HEADER_SIZE}-byte header")
    if data[0] != VERSION:
        raise PacketFormatError(f"unsupported version {data[0]}")
    if data[1] != 0:
        raise PacketFormatError("reserved flags must be 0")
    sequence = int.from_bytes(data[2:6], "big")
    total = int.from_bytes(data[6:10], "big")
    length = int.from_bytes(data[10:12], "big")
    checksum = int.from_bytes(data[12:14], "big")
    if len(data) - HEADER_SIZE != length:
        raise PacketFormatError(f"header declares {length} payload bytes but {len(data) - HEADER_SIZE} follow")
    if total < 1:
        raise PacketFormatError("total must be at least 1")
    return Packet(sequence, total, data[HEADER_SIZE:], checksum)


# --------------------------------------------------------------------------- creation
def create_packets(data: bytes, payload_size: int = DEFAULT_PAYLOAD_SIZE) -> list[Packet]:
    """Split ``data`` into packets of at most ``payload_size`` payload bytes (see module docstring)."""
    data = _as_bytes(data, "data")
    if not _is_int(payload_size):
        raise TypeError("payload_size must be an int")
    if not 1 <= payload_size <= MAX_PAYLOAD_SIZE:
        raise ValueError(f"payload_size must be in 1..{MAX_PAYLOAD_SIZE}, got {payload_size}")
    if not data:
        return [Packet.build(0, 1, b"")]
    total = (len(data) + payload_size - 1) // payload_size
    return [Packet.build(i, total, data[i * payload_size:(i + 1) * payload_size]) for i in range(total)]


# --------------------------------------------------------------------------- diagnosis and reassembly
def _check_packet_list(packets: Any) -> list[Packet]:
    if not isinstance(packets, (list, tuple)):
        raise TypeError("packets must be a list of Packet objects")
    for p in packets:
        if not isinstance(p, Packet):
            raise TypeError("every element of packets must be a Packet")
    return list(packets)


def inspect_packets(packets: Sequence[Packet], allow_identical_duplicates: bool = False) -> dict[str, Any]:
    """Non-raising diagnosis of a received packet set (JSON-compatible).

    Packets that fail their header checksum are reported (``corrupted_*``) but are NOT counted as received
    sequence numbers, totals or duplicates: only packets with valid headers take part in the other fields.

    ``status`` is the highest-priority problem (see the module docstring) or ``"complete"``.
    ``missing`` / ``duplicate_sequences`` / ``invalid_sequences`` are bounded lists; the matching ``*_count``
    fields are exact.
    """
    packets = _check_packet_list(packets)
    report: dict[str, Any] = {
        "received_count": len(packets), "expected_total": None, "totals_seen": [],
        "corrupted_positions": [], "corrupted_count": 0, "invalid_sequences": [], "invalid_sequence_count": 0,
        "duplicate_sequences": [], "duplicate_sequence_count": 0, "conflicting_duplicate_sequences": [],
        "unique_sequence_count": 0, "missing": [], "missing_count": 0, "missing_truncated": False,
        "status": "complete", "complete": False,
    }
    if not packets:
        report["status"] = "no_packets"
        return report
    corrupted = [i for i, p in enumerate(packets) if not p.header_ok()]
    corrupted_count = len(corrupted)
    report["corrupted_positions"] = corrupted[:DETAIL_LIST_LIMIT]
    report["corrupted_count"] = corrupted_count
    # Everything below uses only packets whose header checksum is valid: a packet that fails it cannot be
    # trusted to fill a slot (a receiver discards it), and its corrupted fields must not create bogus
    # duplicates or hide a missing packet.
    usable = [p for p in packets if p.header_ok()]
    totals = sorted({p.total for p in usable})
    report["totals_seen"] = totals[:DETAIL_LIST_LIMIT]
    expected = totals[0] if len(totals) == 1 else None
    report["expected_total"] = expected
    invalid = sorted({p.sequence for p in usable if p.sequence >= p.total})
    report["invalid_sequence_count"] = len(invalid)
    report["invalid_sequences"] = invalid[:DETAIL_LIST_LIMIT]
    by_seq: dict[int, list[bytes]] = {}
    for p in usable:
        by_seq.setdefault(p.sequence, []).append(p.payload)
    dupes = sorted(s for s, payloads in by_seq.items() if len(payloads) > 1)
    conflicting = sorted(s for s in dupes if len(set(by_seq[s])) > 1)
    report["conflicting_duplicate_sequences"] = conflicting[:DETAIL_LIST_LIMIT]
    counted_dupes = conflicting if allow_identical_duplicates else dupes
    report["duplicate_sequence_count"] = len(counted_dupes)
    report["duplicate_sequences"] = counted_dupes[:DETAIL_LIST_LIMIT]
    valid_seqs = {s for s in by_seq if expected is None or s < expected}
    report["unique_sequence_count"] = len(by_seq)
    if expected is not None:
        report["missing_count"] = expected - len(valid_seqs)
        missing: list[int] = []
        i = 0
        while i < expected and len(missing) < DETAIL_LIST_LIMIT:    # bounded: never builds range(expected)
            if i not in by_seq:
                missing.append(i)
            i += 1
        report["missing"] = missing
        report["missing_truncated"] = report["missing_count"] > len(missing)
    if corrupted_count:
        report["status"] = "header_checksum"
    elif len(totals) > 1:
        report["status"] = "inconsistent_total"
    elif invalid:
        report["status"] = "invalid_sequence"
    elif counted_dupes:
        report["status"] = "duplicate"
    elif report["missing_count"]:
        report["status"] = "missing"
    else:
        report["complete"] = True
    return report


def _preview(values: Sequence[int], limit: int = 10) -> str:
    shown = ", ".join(str(v) for v in values[:limit])
    return shown + (", ..." if len(values) > limit else "")


def reassemble_packets(packets: list[Packet], allow_identical_duplicates: bool = False) -> bytes:
    """Rebuild the original bytes from packets received in any order.

    Raises a ``PacketError`` subclass describing the highest-priority problem (see the module docstring).
    With ``allow_identical_duplicates`` a repeated packet whose payload equals the first copy is ignored;
    conflicting copies still raise ``DuplicatePacketsError``.
    """
    packets = _check_packet_list(packets)
    rep = inspect_packets(packets, allow_identical_duplicates)
    status = rep["status"]
    if status == "no_packets":
        raise NoPacketsError("no packets received: an empty message is one packet with an empty payload", received_count=0)
    if status == "header_checksum":
        raise HeaderChecksumError(
            f"{rep['corrupted_count']} packet(s) failed the header checksum (positions: {_preview(rep['corrupted_positions'])})",
            corrupted_positions=rep["corrupted_positions"], corrupted_count=rep["corrupted_count"])
    if status == "inconsistent_total":
        raise InconsistentTotalError(f"packets disagree about the total packet count: {_preview(rep['totals_seen'])}",
                                     totals_seen=rep["totals_seen"])
    if status == "invalid_sequence":
        raise InvalidSequenceError(
            f"sequence number(s) outside 0..{rep['expected_total'] - 1}: {_preview(rep['invalid_sequences'])}",
            invalid_sequences=rep["invalid_sequences"], invalid_sequence_count=rep["invalid_sequence_count"],
            expected_total=rep["expected_total"])
    if status == "duplicate":
        conflicting = bool(rep["conflicting_duplicate_sequences"])
        raise DuplicatePacketsError(
            f"duplicate packet(s) for sequence(s) {_preview(rep['duplicate_sequences'])}"
            + (" with conflicting payloads" if conflicting else ""),
            duplicate_sequences=rep["duplicate_sequences"], duplicate_sequence_count=rep["duplicate_sequence_count"],
            conflicting=conflicting)
    if status == "missing":
        raise MissingPacketsError(
            f"{rep['missing_count']} of {rep['expected_total']} packet(s) missing: {_preview(rep['missing'])}",
            missing=rep["missing"], missing_count=rep["missing_count"], missing_truncated=rep["missing_truncated"],
            expected_total=rep["expected_total"], received_unique=rep["unique_sequence_count"])
    first: dict[int, bytes] = {}
    for p in packets:
        first.setdefault(p.sequence, p.payload)
    return b"".join(first[s] for s in sorted(first))


# --------------------------------------------------------------------------- statistics
def packetization_stats(packets: Sequence[Packet]) -> dict[str, Any]:
    """Overhead accounting: payload bytes versus serialized bytes (headers included)."""
    packets = _check_packet_list(packets)
    payload = sum(p.payload_length for p in packets)
    serialized = sum(p.serialized_size for p in packets)
    return {
        "packet_count": len(packets), "payload_bytes": payload, "header_bytes_total": HEADER_SIZE * len(packets),
        "serialized_bytes_total": serialized, "header_size": HEADER_SIZE,
        "overhead_percent": ((serialized - payload) / serialized * 100) if serialized else None,
        "efficiency_percent": (payload / serialized * 100) if serialized else None,
        "largest_payload": max((p.payload_length for p in packets), default=0),
        "last_payload": packets[-1].payload_length if packets else 0,
    }


# --------------------------------------------------------------------------- experiment helpers
def corrupt_header(packet: Packet, bit: int = 0) -> Packet:
    """Flip one bit of the SEQUENCE field but keep the old checksum (a header-corruption experiment)."""
    if not isinstance(packet, Packet):
        raise TypeError("packet must be a Packet")
    if not _is_int(bit) or not 0 <= bit < 32:
        raise ValueError("bit must be in 0..31")
    return replace(packet, sequence=packet.sequence ^ (1 << bit))


def drop_packets(packets: Sequence[Packet], loss_probability: float, seed: Optional[int] = None
                 ) -> tuple[list[Packet], list[int]]:
    """Lose each packet independently with probability ``loss_probability`` (reproducible with ``seed``).

    Returns (surviving packets in the original order, sorted list of the LIST POSITIONS that were dropped).
    """
    packets = _check_packet_list(packets)
    if isinstance(loss_probability, bool) or not isinstance(loss_probability, (int, float)):
        raise TypeError("loss_probability must be a number")
    if not 0.0 <= loss_probability <= 1.0:                # also rejects NaN
        raise ValueError("loss_probability must satisfy 0 <= p <= 1")
    rng = random.Random(seed)
    kept, dropped = [], []
    for i, p in enumerate(packets):
        if rng.random() < loss_probability:
            dropped.append(i)
        else:
            kept.append(p)
    return kept, dropped

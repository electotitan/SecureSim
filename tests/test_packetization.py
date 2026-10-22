"""Checkpoint 7: packetization tests."""
import random
from dataclasses import replace

import pytest

from backend.pipeline import packetization as pk
from backend.pipeline.packetization import (
    DETAIL_LIST_LIMIT, HEADER_SIZE, DuplicatePacketsError, HeaderChecksumError, InconsistentTotalError,
    InvalidSequenceError, MissingPacketsError, NoPacketsError, Packet, PacketError, PacketFormatError,
    corrupt_header, create_packets, drop_packets, inspect_packets, internet_checksum, packet_from_bytes,
    packetization_stats, reassemble_packets,
)


def rand_bytes(rng, n):
    return bytes(rng.randrange(256) for _ in range(n))


def ref_checksum(data: bytes) -> int:
    """Independent reference: big-integer word sum with end-around carry folding."""
    if len(data) % 2:
        data += b"\x00"
    s = sum(int.from_bytes(data[i:i + 2], "big") for i in range(0, len(data), 2))
    while s > 0xFFFF:
        s = (s & 0xFFFF) + (s >> 16)
    return ~s & 0xFFFF


# ================================================================== Internet checksum (RFC 1071)
def test_rfc1071_worked_example():
    # RFC 1071 section 3: words 0001 f203 f4f5 f6f7 sum to ddf2 (after folding); the checksum is its complement 220d.
    assert internet_checksum(bytes.fromhex("0001f203f4f5f6f7")) == 0x220D


def test_ipv4_header_example_from_the_literature():
    # IPv4 header 4500 0073 0000 4000 4011 [0000] c0a8 0001 c0a8 00c7 -> checksum b861
    hdr = bytes.fromhex("45000073000040004011" "0000" "c0a80001c0a800c7")
    assert internet_checksum(hdr) == 0xB861
    assert internet_checksum(hdr[:10] + bytes.fromhex("b861") + hdr[12:]) == 0      # receiver check


def test_checksum_edge_cases():
    assert internet_checksum(b"") == 0xFFFF
    assert internet_checksum(b"\x00" * 8) == 0xFFFF
    assert internet_checksum(b"\xff\xff") == 0x0000
    assert internet_checksum(b"\x01") == internet_checksum(b"\x01\x00") == 0xFEFF       # odd length is zero-padded
    assert internet_checksum(b"\xff\xff\x00\x01") == 0xFFFE                              # carry wraps around: ffff+0001 -> 0001


def test_checksum_matches_independent_reference():
    rng = random.Random(1)
    for _ in range(2000):
        data = rand_bytes(rng, rng.randint(0, 80))
        assert internet_checksum(data) == ref_checksum(data)


def test_appending_the_checksum_makes_the_whole_sum_zero():
    rng = random.Random(2)
    for _ in range(500):
        data = rand_bytes(rng, rng.randrange(0, 60, 2))                 # even length so the checksum is word aligned
        assert internet_checksum(data + internet_checksum(data).to_bytes(2, "big")) == 0


def test_checksum_is_invariant_under_word_reordering_a_documented_weakness():
    a = bytes.fromhex("1234 5678 9abc def0".replace(" ", ""))
    b = bytes.fromhex("9abc 1234 def0 5678".replace(" ", ""))
    assert a != b and internet_checksum(a) == internet_checksum(b)


def test_checksum_misses_two_cancelling_errors_a_documented_weakness():
    a = bytes.fromhex("0010 0020".replace(" ", ""))
    b = bytes.fromhex("0011 001f".replace(" ", ""))                       # +1 and -1
    assert internet_checksum(a) == internet_checksum(b)


def test_checksum_detects_every_single_bit_error():
    rng = random.Random(3)
    for _ in range(40):
        data = rand_bytes(rng, rng.randrange(2, 40, 2))
        base = internet_checksum(data)
        for bit in range(8 * len(data)):
            flipped = bytearray(data)
            flipped[bit // 8] ^= 1 << (bit % 8)
            assert internet_checksum(bytes(flipped)) != base


def test_checksum_type_errors():
    for bad in ("abc", 5, None, [1, 2]):
        with pytest.raises(TypeError):
            internet_checksum(bad)
    assert internet_checksum(bytearray(b"ab")) == internet_checksum(b"ab")


# ================================================================== packet format
def test_constants_and_layout():
    assert HEADER_SIZE == 14 and pk.DEFAULT_PAYLOAD_SIZE == 64 and pk.VERSION == 1


def test_hand_derived_wire_bytes():
    # seq 0, total 2, payload "AB": words 0100 0000 0000 0000 0002 0002 [0000] -> sum 0104 -> checksum fefb
    p = Packet.build(0, 2, b"AB")
    assert p.checksum == 0xFEFB
    assert p.to_bytes() == bytes.fromhex("0100" "00000000" "00000002" "0002" "fefb") + b"AB"
    assert p.header_bytes().hex() == "01000000000000000002" "0002" "fefb"
    assert p.serialized_size == 16 and p.payload_length == 2 and p.header_ok()


def test_serialized_size_is_header_plus_payload_not_payload_size():
    for n in (0, 1, 63, 64, 1000):
        p = Packet.build(0, 1, b"x" * n)
        assert p.serialized_size == HEADER_SIZE + n != n


def test_header_fields_dict():
    p = Packet.build(7, 9, b"hello")
    assert p.header_fields() == {"version": 1, "flags": 0, "sequence": 7, "total": 9, "payload_length": 5,
                                 "checksum": p.checksum}


def test_serialize_parse_round_trip_random():
    rng = random.Random(4)
    for _ in range(300):
        total = rng.randint(1, 2 ** 32 - 1)
        p = Packet.build(rng.randrange(0, total), total, rand_bytes(rng, rng.randint(0, 200)))
        assert packet_from_bytes(p.to_bytes()) == p and packet_from_bytes(p.to_bytes()).header_ok()


def test_extreme_field_values_round_trip():
    for seq, total in ((0, 1), (2 ** 32 - 1, 2 ** 32 - 1), (0, 2 ** 32 - 1)):
        p = Packet.build(seq, total, b"z")
        assert packet_from_bytes(p.to_bytes()) == p and p.header_ok()
    big = Packet.build(0, 1, bytes(65535))
    assert packet_from_bytes(big.to_bytes()) == big and big.serialized_size == 65535 + 14


@pytest.mark.parametrize("raw, fragment", [
    (b"", "shorter than"), (b"\x01" * 13, "shorter than"),
    (bytes.fromhex("0200" "00000000" "00000001" "0000" "0000"), "unsupported version"),
    (bytes.fromhex("0101" "00000000" "00000001" "0000" "0000"), "flags"),
    (bytes.fromhex("0100" "00000000" "00000001" "0005" "0000") + b"abc", "declares 5"),       # truncated payload
    (bytes.fromhex("0100" "00000000" "00000001" "0001" "0000") + b"abc", "declares 1"),       # trailing bytes
    (bytes.fromhex("0100" "00000000" "00000000" "0000" "0000"), "total"),
])
def test_parse_rejects_malformed_bytes(raw, fragment):
    with pytest.raises(PacketFormatError, match=fragment):
        packet_from_bytes(raw)


def test_parse_type_errors():
    for bad in ("abc", 5, None):
        with pytest.raises(TypeError):
            packet_from_bytes(bad)


def test_parse_does_not_verify_the_checksum_so_bad_packets_stay_inspectable():
    raw = bytearray(Packet.build(1, 4, b"abc").to_bytes())
    raw[13] ^= 0xFF
    p = packet_from_bytes(bytes(raw))
    assert p.sequence == 1 and p.header_ok() is False


@pytest.mark.parametrize("kwargs, exc", [
    (dict(sequence=-1, total=1, payload=b"", checksum=0), ValueError),
    (dict(sequence=2 ** 32, total=1, payload=b"", checksum=0), ValueError),
    (dict(sequence=0, total=0, payload=b"", checksum=0), ValueError),
    (dict(sequence=0, total=2 ** 32, payload=b"", checksum=0), ValueError),
    (dict(sequence=0, total=1, payload=bytes(65536), checksum=0), ValueError),
    (dict(sequence=0, total=1, payload=b"", checksum=-1), ValueError),
    (dict(sequence=0, total=1, payload=b"", checksum=65536), ValueError),
    (dict(sequence=0.0, total=1, payload=b"", checksum=0), TypeError),
    (dict(sequence=True, total=1, payload=b"", checksum=0), TypeError),
    (dict(sequence=0, total="1", payload=b"", checksum=0), TypeError),
    (dict(sequence=0, total=1, payload="abc", checksum=0), TypeError),
    (dict(sequence=0, total=1, payload=bytearray(b"a"), checksum=0), TypeError),
    (dict(sequence=0, total=1, payload=b"", checksum=1.5), TypeError),
])
def test_packet_field_validation(kwargs, exc):
    with pytest.raises(exc):
        Packet(**kwargs)


def test_packet_build_validation_and_immutability():
    for args, exc in (((0, 0, b""), ValueError), ((-1, 1, b""), ValueError), ((0, 1, "x"), TypeError),
                      ((0.5, 1, b""), TypeError), ((0, 1, bytes(65536)), ValueError)):
        with pytest.raises(exc):
            Packet.build(*args)
    p = Packet.build(0, 1, b"x")
    with pytest.raises(Exception):
        p.sequence = 5
    assert Packet.build(0, 1, bytearray(b"x")) == p


# ================================================================== header integrity status
def test_every_created_packet_has_a_valid_header():
    rng = random.Random(5)
    for _ in range(100):
        for p in create_packets(rand_bytes(rng, rng.randint(0, 500)), rng.randint(1, 100)):
            assert p.header_ok()


def test_changing_any_header_field_without_recomputing_the_checksum_is_detected():
    p = Packet.build(3, 10, b"payload!")
    assert not replace(p, sequence=4).header_ok()
    assert not replace(p, total=11).header_ok()
    assert not replace(p, payload=b"short").header_ok()                   # payload_length is part of the header
    assert not replace(p, payload=b"much longer payload").header_ok()
    assert not replace(p, checksum=p.checksum ^ 1).header_ok()


def test_every_single_bit_flip_of_the_serialized_header_is_detected():
    rng = random.Random(6)
    packets = [Packet.build(0, 1, b""), Packet.build(2, 5, b"abc"), Packet.build(2 ** 32 - 2, 2 ** 32 - 1, b"z" * 20),
               Packet.build(rng.randrange(1000), 1000, rand_bytes(rng, 64))]
    detected = 0
    for p in packets:
        raw = p.to_bytes()
        for bit in range(8 * HEADER_SIZE):                                  # all 112 header bits
            flipped = bytearray(raw)
            flipped[bit // 8] ^= 1 << (7 - bit % 8)
            try:
                parsed = packet_from_bytes(bytes(flipped))
            except PacketFormatError:
                detected += 1                                               # structurally invalid
                continue
            assert parsed.header_ok() is False, (p, bit)
            detected += 1
    assert detected == 4 * 112


def test_header_checksum_does_not_protect_the_payload_a_documented_limit():
    p = Packet.build(0, 1, b"ORIGINAL")
    tampered = replace(p, payload=b"TAMPERED")                              # same length, different bytes
    assert tampered.header_ok() is True                                      # undetected by the header checksum
    assert reassemble_packets([tampered]) == b"TAMPERED"                     # ... so CRC-32 is needed next


def test_word_swap_in_header_is_invisible_to_the_checksum_but_caught_by_consistency_checks():
    packets = create_packets(b"x" * 50, 10)                                  # 5 packets, total 5
    victim = packets[3]                                                      # sequence 3, total 5
    swapped = replace(victim, sequence=5, total=3)                           # swap the low words of the two fields
    assert swapped.checksum == victim.checksum and swapped.header_ok() is True
    with pytest.raises(PacketError) as info:
        reassemble_packets(packets[:3] + [swapped, packets[4]])
    assert info.value.kind in ("inconsistent_total", "invalid_sequence")


def test_cancelling_header_errors_are_invisible_to_the_checksum():
    p = Packet.build(2, 5, b"abc")
    forged = replace(p, sequence=3, total=4)                                 # +1 and -1 cancel in the sum
    assert forged.header_ok() is True and forged.sequence != p.sequence


# ================================================================== create_packets
def test_empty_message_is_one_empty_packet():
    ps = create_packets(b"")
    assert len(ps) == 1 and (ps[0].sequence, ps[0].total, ps[0].payload) == (0, 1, b"") and ps[0].header_ok()
    assert reassemble_packets(ps) == b""


@pytest.mark.parametrize("n, size, count, last", [
    (1, 64, 1, 1), (63, 64, 1, 63), (64, 64, 1, 64), (65, 64, 2, 1), (128, 64, 2, 64), (130, 64, 3, 2),
    (10, 1, 10, 1), (10, 3, 4, 1), (10, 10, 1, 10), (10, 100, 1, 10), (1000, 64, 16, 40), (999, 333, 3, 333),
])
def test_chunking_arithmetic(n, size, count, last):
    data = bytes(range(256)) * 4
    data = data[:n]
    ps = create_packets(data, size)
    assert len(ps) == count and ps[-1].payload_length == last
    assert all(p.payload_length == size for p in ps[:-1])
    assert [p.sequence for p in ps] == list(range(count)) and {p.total for p in ps} == {count}
    assert b"".join(p.payload for p in ps) == data
    assert all(p.payload_length > 0 for p in ps)                              # never an empty trailing packet


def test_default_payload_size_is_64():
    assert [p.payload_length for p in create_packets(b"a" * 130)] == [64, 64, 2]


@pytest.mark.parametrize("size", [0, -1, -64, 65536, 10 ** 9])
def test_invalid_payload_sizes_rejected(size):
    with pytest.raises(ValueError, match="payload_size"):
        create_packets(b"abc", size)


@pytest.mark.parametrize("size", [64.0, "64", None, True])
def test_non_int_payload_size_rejected(size):
    with pytest.raises(TypeError):
        create_packets(b"abc", size)


def test_boundary_payload_sizes_accepted():
    assert len(create_packets(b"abc", 1)) == 3
    assert len(create_packets(bytes(70000), 65535)) == 2
    assert create_packets(bytes(70000), 65535)[0].payload_length == 65535


@pytest.mark.parametrize("bad", ["abc", 5, None, [1, 2]])
def test_create_packets_rejects_non_bytes(bad):
    with pytest.raises(TypeError):
        create_packets(bad)


def test_create_packets_accepts_bytearray_and_is_deterministic():
    assert create_packets(bytearray(b"hello world"), 4) == create_packets(b"hello world", 4)


# ================================================================== reassemble: round trips and order
def test_round_trip_many_sizes_and_payload_sizes():
    rng = random.Random(7)
    for _ in range(300):
        data = rand_bytes(rng, rng.randint(0, 800))
        size = rng.randint(1, 200)
        assert reassemble_packets(create_packets(data, size)) == data


def test_reassembly_preserves_order_regardless_of_arrival_order():
    rng = random.Random(8)
    data = rand_bytes(rng, 1000)
    ps = create_packets(data, 37)
    for _ in range(30):
        shuffled = ps[:]
        rng.shuffle(shuffled)
        assert reassemble_packets(shuffled) == data
    assert reassemble_packets(list(reversed(ps))) == data


def test_reassembly_does_not_mutate_its_input_and_accepts_tuples():
    ps = create_packets(b"x" * 100, 10)
    shuffled = ps[::-1]
    snapshot = list(shuffled)
    reassemble_packets(shuffled)
    assert shuffled == snapshot
    assert reassemble_packets(tuple(ps)) == b"x" * 100


@pytest.mark.parametrize("bad", ["abc", None, 5, {"a": 1}])
def test_reassemble_type_errors_for_container(bad):
    with pytest.raises(TypeError):
        reassemble_packets(bad)


def test_reassemble_type_errors_for_elements():
    with pytest.raises(TypeError):
        reassemble_packets([Packet.build(0, 1, b"a"), b"raw bytes"])
    with pytest.raises(TypeError):
        inspect_packets([5])


# ================================================================== reassemble: failure detection
def test_no_packets():
    with pytest.raises(NoPacketsError) as info:
        reassemble_packets([])
    assert info.value.kind == "no_packets" and info.value.details == {"received_count": 0}


@pytest.mark.parametrize("missing", [[0], [4], [2], [1, 3], [0, 1, 2, 3], [0, 4]])
def test_missing_packets_detected_with_exact_sequence_numbers(missing):
    ps = create_packets(b"y" * 50, 10)                                        # sequences 0..4
    kept = [p for p in ps if p.sequence not in missing]
    if not kept:
        kept = []
        with pytest.raises(NoPacketsError):
            reassemble_packets(kept)
        return
    with pytest.raises(MissingPacketsError) as info:
        reassemble_packets(kept)
    d = info.value.details
    assert d["missing"] == sorted(missing) and d["missing_count"] == len(missing)
    assert d["expected_total"] == 5 and d["received_unique"] == 5 - len(missing) and d["missing_truncated"] is False


def test_missing_last_packet_is_detected_not_silently_accepted():
    ps = create_packets(b"abcdefghij", 3)                                     # 4 packets
    with pytest.raises(MissingPacketsError) as info:
        reassemble_packets(ps[:-1])
    assert info.value.details["missing"] == [3]


def test_single_packet_message_cannot_be_missing_a_packet():
    assert reassemble_packets(create_packets(b"short")) == b"short"


def test_duplicate_packets_detected():
    ps = create_packets(b"z" * 50, 10)
    with pytest.raises(DuplicatePacketsError) as info:
        reassemble_packets(ps + [ps[2]])
    d = info.value.details
    assert d["duplicate_sequences"] == [2] and d["conflicting"] is False and d["duplicate_sequence_count"] == 1
    with pytest.raises(DuplicatePacketsError) as info:
        reassemble_packets(ps + [ps[0], ps[0], ps[4]])
    assert info.value.details["duplicate_sequences"] == [0, 4]


def test_identical_duplicates_can_be_tolerated_on_request():
    data = bytes(range(100))
    ps = create_packets(data, 10)
    noisy = ps + [ps[3], ps[3], ps[0]]
    assert reassemble_packets(noisy, allow_identical_duplicates=True) == data
    assert inspect_packets(noisy, allow_identical_duplicates=True)["status"] == "complete"
    assert inspect_packets(noisy)["status"] == "duplicate"


def test_conflicting_duplicates_are_never_tolerated():
    ps = create_packets(b"a" * 30, 10)
    evil = Packet.build(1, 3, b"b" * 10)                                      # same sequence, valid header, different payload
    with pytest.raises(DuplicatePacketsError) as info:
        reassemble_packets(ps + [evil], allow_identical_duplicates=True)
    assert info.value.details["conflicting"] is True and info.value.details["duplicate_sequences"] == [1]


def test_inconsistent_total_counts_detected():
    ps = create_packets(b"q" * 40, 10)                                        # total 4
    other = Packet.build(1, 5, b"q" * 10)
    with pytest.raises(InconsistentTotalError) as info:
        reassemble_packets([ps[0], other, ps[2], ps[3]])
    assert info.value.details["totals_seen"] == [4, 5]


def test_sequence_outside_declared_total_detected():
    ps = create_packets(b"r" * 30, 10)                                        # total 3
    rogue = Packet.build(7, 3, b"x")
    with pytest.raises(InvalidSequenceError) as info:
        reassemble_packets(ps + [rogue])
    d = info.value.details
    assert d["invalid_sequences"] == [7] and d["expected_total"] == 3 and d["invalid_sequence_count"] == 1
    with pytest.raises(InvalidSequenceError):
        reassemble_packets([Packet.build(3, 3, b"x")])                         # sequence == total is already out of range


def test_header_checksum_failure_detected():
    ps = create_packets(b"s" * 30, 10)
    bad = replace(ps[1], sequence=2)                                          # corrupt, checksum kept
    with pytest.raises(HeaderChecksumError) as info:
        reassemble_packets([ps[0], bad, ps[2]])
    assert info.value.details == {"corrupted_positions": [1], "corrupted_count": 1}
    with pytest.raises(HeaderChecksumError) as info:
        reassemble_packets([corrupt_header(p) for p in ps])
    assert info.value.details["corrupted_count"] == 3


def test_checksum_failure_takes_priority_over_every_other_problem():
    ps = create_packets(b"t" * 50, 10)
    messy = [corrupt_header(ps[0]), ps[1], ps[1], Packet.build(1, 99, b"x"), Packet.build(77, 5, b"x")]   # many problems at once
    with pytest.raises(HeaderChecksumError):
        reassemble_packets(messy)


def test_error_priority_total_then_sequence_then_duplicate_then_missing():
    ps = create_packets(b"u" * 50, 10)                                        # total 5, seq 0..4
    with pytest.raises(InconsistentTotalError):
        reassemble_packets([ps[0], ps[0], Packet.build(9, 6, b"x")])           # dup + invalid + total mismatch
    with pytest.raises(InvalidSequenceError):
        reassemble_packets([ps[0], ps[0], Packet.build(9, 5, b"x")])           # dup + invalid sequence
    with pytest.raises(DuplicatePacketsError):
        reassemble_packets([ps[0], ps[0], ps[2]])                              # dup + missing
    with pytest.raises(MissingPacketsError):
        reassemble_packets([ps[0], ps[2]])


def test_error_class_hierarchy_and_kinds():
    kinds = {NoPacketsError: "no_packets", HeaderChecksumError: "header_checksum", InconsistentTotalError: "inconsistent_total",
             InvalidSequenceError: "invalid_sequence", DuplicatePacketsError: "duplicate", MissingPacketsError: "missing",
             PacketFormatError: "format"}
    for cls, kind in kinds.items():
        assert issubclass(cls, PacketError) and issubclass(cls, ValueError) and cls.kind == kind


def test_forged_huge_total_is_reported_without_exhausting_memory():
    forged = [Packet.build(0, 2 ** 32 - 1, b"x"), Packet.build(5, 2 ** 32 - 1, b"y")]
    with pytest.raises(MissingPacketsError) as info:
        reassemble_packets(forged)
    d = info.value.details
    assert d["missing_count"] == 2 ** 32 - 1 - 2 and len(d["missing"]) == DETAIL_LIST_LIMIT
    assert d["missing_truncated"] is True and d["missing"][:5] == [1, 2, 3, 4, 6]
    assert "..." in str(info.value) and len(str(info.value)) < 300


def test_many_invalid_sequences_are_bounded_in_details():
    ps = [Packet.build(10 + i, 5, b"x") for i in range(3000)]
    with pytest.raises(InvalidSequenceError) as info:
        reassemble_packets(ps)
    assert info.value.details["invalid_sequence_count"] == 3000 and len(info.value.details["invalid_sequences"]) == DETAIL_LIST_LIMIT


# ================================================================== inspect_packets (non-raising diagnosis)
def test_inspect_complete_set():
    rep = inspect_packets(create_packets(b"v" * 55, 10))
    assert rep["status"] == "complete" and rep["complete"] is True
    assert (rep["received_count"], rep["expected_total"], rep["missing_count"], rep["unique_sequence_count"]) == (6, 6, 0, 6)
    assert rep["corrupted_count"] == 0 and rep["totals_seen"] == [6]


def test_inspect_reports_every_problem_independently_without_raising():
    ps = create_packets(b"w" * 60, 10)                                        # total 6
    rep = inspect_packets([ps[0], ps[0], corrupt_header(ps[2]), ps[4], Packet.build(9, 6, b"x")])
    assert rep["status"] == "header_checksum" and rep["complete"] is False
    assert rep["corrupted_positions"] == [2] and rep["duplicate_sequences"] == [0]
    assert rep["invalid_sequences"] == [9] and rep["expected_total"] == 6
    # the corrupted packet (claimed sequence 3) is discarded, so 2 and 3 count as missing along with 1 and 5
    assert rep["missing"] == [1, 2, 3, 5] and rep["missing_count"] == 4 and rep["unique_sequence_count"] == 3


def test_inspect_missing_accounts_only_for_in_range_sequences():
    rep = inspect_packets([Packet.build(0, 3, b"a"), Packet.build(1, 3, b"b"), Packet.build(8, 3, b"c")])
    assert rep["status"] == "invalid_sequence"
    assert rep["missing"] == [2] and rep["missing_count"] == 1


def test_inspect_inconsistent_totals_has_no_expected_total_and_no_missing_claims():
    rep = inspect_packets([Packet.build(0, 3, b"a"), Packet.build(1, 4, b"b")])
    assert rep["status"] == "inconsistent_total" and rep["expected_total"] is None
    assert rep["missing"] == [] and rep["missing_count"] == 0 and rep["totals_seen"] == [3, 4]


def test_inspect_empty():
    rep = inspect_packets([])
    assert rep["status"] == "no_packets" and rep["received_count"] == 0 and rep["complete"] is False


def test_inspect_is_json_compatible():
    import json
    json.dumps(inspect_packets(create_packets(b"x" * 40, 7)[:-2]))


def test_inspect_status_matches_what_reassemble_raises():
    ps = create_packets(b"m" * 50, 10)
    scenarios = [
        ([], NoPacketsError), ([corrupt_header(ps[0])] + ps[1:], HeaderChecksumError),
        (ps[:2] + [Packet.build(2, 9, b"x")], InconsistentTotalError), (ps + [Packet.build(8, 5, b"x")], InvalidSequenceError),
        (ps + [ps[1]], DuplicatePacketsError), (ps[:-1], MissingPacketsError),
    ]
    for received, exc in scenarios:
        with pytest.raises(exc) as info:
            reassemble_packets(received)
        assert inspect_packets(received)["status"] == info.value.kind


# ================================================================== statistics
def test_stats_known_values():
    s = packetization_stats(create_packets(b"a" * 130, 64))
    assert (s["packet_count"], s["payload_bytes"], s["header_bytes_total"], s["serialized_bytes_total"]) == (3, 130, 42, 172)
    assert s["overhead_percent"] == pytest.approx(42 / 172 * 100) and s["efficiency_percent"] == pytest.approx(130 / 172 * 100)
    assert (s["largest_payload"], s["last_payload"], s["header_size"]) == (64, 2, 14)


def test_stats_empty_message_has_zero_efficiency_and_full_overhead():
    s = packetization_stats(create_packets(b""))
    assert (s["packet_count"], s["payload_bytes"], s["serialized_bytes_total"]) == (1, 0, 14)
    assert s["efficiency_percent"] == 0.0 and s["overhead_percent"] == 100.0


def test_stats_overhead_shrinks_with_larger_payloads():
    data = b"d" * 1000
    eff = [packetization_stats(create_packets(data, size))["efficiency_percent"] for size in (8, 32, 128, 512)]
    assert eff == sorted(eff) and eff[0] < eff[-1]


def test_stats_of_no_packets():
    s = packetization_stats([])
    assert s["packet_count"] == 0 and s["efficiency_percent"] is None and s["overhead_percent"] is None


# ================================================================== experiment helpers
def test_corrupt_header_flips_one_sequence_bit_and_keeps_the_checksum():
    p = Packet.build(4, 9, b"abc")
    c = corrupt_header(p, 0)
    assert c.sequence == 5 and c.checksum == p.checksum and c.payload == p.payload and not c.header_ok()
    assert corrupt_header(p, 3).sequence == 4 ^ 8
    for bad in (-1, 32, 100):
        with pytest.raises(ValueError):
            corrupt_header(p, bad)
    for bad in (1.5, "0", True):
        with pytest.raises(ValueError if bad is True else (TypeError, ValueError)):
            corrupt_header(p, bad)
    with pytest.raises(TypeError):
        corrupt_header(b"raw")


def test_drop_packets_extremes_and_order():
    ps = create_packets(b"e" * 100, 10)
    assert drop_packets(ps, 0.0, 1) == (ps, [])
    kept, dropped = drop_packets(ps, 1.0, 1)
    assert kept == [] and dropped == list(range(10))
    kept, dropped = drop_packets(ps, 0.5, 3)
    assert [p for i, p in enumerate(ps) if i not in dropped] == kept and dropped == sorted(dropped)
    assert len(kept) + len(dropped) == 10


def test_drop_packets_is_reproducible_with_a_seed():
    ps = create_packets(b"f" * 2000, 10)
    assert drop_packets(ps, 0.3, 42) == drop_packets(ps, 0.3, 42)
    assert drop_packets(ps, 0.3, 42)[1] != drop_packets(ps, 0.3, 43)[1]


def test_drop_rate_matches_the_requested_probability_statistically():
    ps = create_packets(b"g" * 20000, 10)                                     # 2000 packets
    for p in (0.1, 0.3, 0.7):
        _, dropped = drop_packets(ps, p, 11)
        assert abs(len(dropped) / 2000 - p) < 0.04


def test_drop_packets_loss_is_detected_as_missing_by_reassembly():
    data = bytes(range(200))
    ps = create_packets(data, 10)
    kept, dropped = drop_packets(ps, 0.3, 5)
    assert dropped
    with pytest.raises(MissingPacketsError) as info:
        reassemble_packets(kept)
    assert info.value.details["missing"] == dropped or kept == []   # positions equal sequences for an unshuffled set


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf"), 2])
def test_drop_packets_rejects_invalid_probabilities(p):
    with pytest.raises(ValueError):
        drop_packets(create_packets(b"x"), p)


@pytest.mark.parametrize("p", ["0.5", None, True, [0.5]])
def test_drop_packets_rejects_non_numeric_probabilities(p):
    with pytest.raises(TypeError):
        drop_packets(create_packets(b"x"), p)


def test_drop_packets_empty_input_and_does_not_mutate():
    assert drop_packets([], 0.5, 1) == ([], [])
    ps = create_packets(b"x" * 30, 10)
    snapshot = list(ps)
    drop_packets(ps, 0.5, 1)
    assert ps == snapshot


# ================================================================== independence
def test_module_uses_no_forbidden_libraries():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "backend" / "pipeline" / "packetization.py").read_text()
    for banned in ("zlib", "crcmod", "import struct", "from struct", "Crypto", "hashlib", "binascii"):
        assert banned not in src


def test_corrupted_packets_are_discarded_not_counted_as_received_sequences():
    ps = create_packets(b"n" * 50, 10)                                        # total 5
    # corrupting packet 0 flips its sequence to 1, colliding with the genuine packet 1
    rep = inspect_packets([corrupt_header(ps[0])] + ps[1:])
    assert rep["status"] == "header_checksum" and rep["corrupted_positions"] == [0] and rep["corrupted_count"] == 1
    assert rep["duplicate_sequences"] == [] and rep["duplicate_sequence_count"] == 0       # no bogus duplicate
    assert rep["missing"] == [0] and rep["missing_count"] == 1                              # the real packet 0 is absent
    assert rep["unique_sequence_count"] == 4 and rep["received_count"] == 5


def test_corrupted_total_does_not_poison_the_totals_or_expected_count():
    ps = create_packets(b"o" * 30, 10)                                        # total 3
    bad = replace(ps[1], total=99)                                            # corrupt total, checksum kept
    rep = inspect_packets([ps[0], bad, ps[2]])
    assert rep["status"] == "header_checksum" and rep["totals_seen"] == [3] and rep["expected_total"] == 3
    assert rep["missing"] == [1] and rep["corrupted_positions"] == [1]


def test_all_packets_corrupted_leaves_nothing_usable():
    ps = create_packets(b"p" * 30, 10)
    rep = inspect_packets([corrupt_header(p) for p in ps])
    assert rep["status"] == "header_checksum" and rep["corrupted_count"] == 3
    assert rep["totals_seen"] == [] and rep["expected_total"] is None and rep["unique_sequence_count"] == 0
    assert rep["missing"] == [] and rep["missing_count"] == 0


def test_corrupt_header_rejects_bad_bit_positions_with_a_specific_message():
    p = Packet.build(4, 9, b"abc")
    for bad in (-1, 32, 33, 100):
        with pytest.raises(ValueError, match="bit must be in 0..31"):
            corrupt_header(p, bad)
    assert corrupt_header(p, 31).sequence == 4 ^ (1 << 31) and not corrupt_header(p, 31).header_ok()

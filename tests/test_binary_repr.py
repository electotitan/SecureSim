"""Checkpoint 3 (Stage 2): binary representation tests."""
import random

import pytest

from backend.pipeline.binary_repr import (
    binary_to_bytes, binary_to_text, bytes_to_binary, describe_characters,
    text_to_binary, visualize_bits,
)


def test_ascii_known_values():
    assert text_to_binary("A") == "01000001"
    assert text_to_binary("Hi") == "0100100001101001"
    assert text_to_binary("") == ""


def test_multibyte_known_values():
    assert text_to_binary("é") == "1100001110101001"                  # C3 A9, not E9
    assert text_to_binary("€") == "111000101000001010101100"          # E2 82 AC
    assert text_to_binary("😀") == "11110000100111111001100010000000"  # F0 9F 98 80


def test_every_byte_is_exactly_eight_bits():
    for text in ("", "a", "é€😀", "hello world", "\x00\x01"):
        bits = text_to_binary(text)
        assert len(bits) == 8 * len(text.encode("utf-8"))
        assert set(bits) <= {"0", "1"}


def test_all_256_byte_values_round_trip_through_bits():
    data = bytes(range(256))
    bits = bytes_to_binary(data)
    assert len(bits) == 2048
    assert binary_to_bytes(bits) == data


def test_round_trip_random_unicode():
    rng = random.Random(11)
    pool = [chr(c) for c in list(range(0, 128)) + list(range(0x80, 0x800, 7)) +
            list(range(0x800, 0xD800, 211)) + list(range(0xE000, 0x10000, 331)) +
            list(range(0x10000, 0x110000, 4001))]
    for _ in range(200):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 60)))
        assert binary_to_text(text_to_binary(text)) == text


def test_empty_round_trip():
    assert binary_to_text("") == ""
    assert binary_to_bytes("") == b""
    assert visualize_bits("") == []


@pytest.mark.parametrize("bad", [
    "0100000a", "0100 0001", "2", "０１",
    # 8-character inputs that Python's int(x, 2) would silently ACCEPT, so only the explicit
    # character check (not the length check or int()) can reject them:
    "０１０００００１", "0100000 ", "+1000001", "-1000001", "0b100000", "0100_001",
])
def test_rejects_non_binary_characters(bad):
    with pytest.raises(ValueError):
        binary_to_text(bad)


@pytest.mark.parametrize("bad", ["0", "0100000", "010000011", "1" * 15])
def test_requires_complete_bytes(bad):
    with pytest.raises(ValueError, match="multiple of 8"):
        binary_to_text(bad)
    with pytest.raises(ValueError, match="multiple of 8"):
        visualize_bits(bad)


@pytest.mark.parametrize("data", [
    b"\xff", b"\x80", b"\xc0\x80", b"\xc3", b"\xe2\x82", b"\xf0\x9f\x98", b"\xed\xa0\x80",
])
def test_invalid_utf8_rejected(data):
    # lone 0xFF, lone continuation, overlong C0 80, truncated sequences, UTF-8 encoded surrogate
    with pytest.raises(ValueError, match="UTF-8"):
        binary_to_text(bytes_to_binary(data))


def test_lone_surrogate_text_rejected():
    with pytest.raises(ValueError, match="UTF-8"):
        text_to_binary("\ud800")
    with pytest.raises(ValueError):
        describe_characters("a\udfff")


def test_type_errors():
    with pytest.raises(TypeError):
        text_to_binary(b"abc")
    with pytest.raises(TypeError):
        binary_to_text(1010)
    with pytest.raises(TypeError):
        bytes_to_binary("abc")


def test_visualize_bits_ascii():
    grid = visualize_bits(text_to_binary("Hi"))
    assert [g["index"] for g in grid] == [0, 1]
    assert grid[0] == {
        "index": 0, "binary": "01001000", "bits": [0, 1, 0, 0, 1, 0, 0, 0], "hex": "48",
        "decimal": 72, "role": "ascii", "character": "H", "char_index": 0, "char_start": True,
    }
    assert grid[1]["character"] == "i" and grid[1]["decimal"] == 105 and grid[1]["hex"] == "69"


def test_visualize_bits_multibyte_roles_and_characters():
    grid = visualize_bits(text_to_binary("a€😀"))
    assert [g["role"] for g in grid] == [
        "ascii", "lead-3", "continuation", "continuation",
        "lead-4", "continuation", "continuation", "continuation"]
    assert [g["character"] for g in grid] == ["a", "€", "€", "€", "😀", "😀", "😀", "😀"]
    assert [g["char_index"] for g in grid] == [0, 1, 1, 1, 2, 2, 2, 2]
    assert [g["char_start"] for g in grid] == [True, True, False, False, True, False, False, False]


def test_visualize_bits_non_utf8_stream_has_no_characters():
    grid = visualize_bits(bytes_to_binary(b"\xff\x00\x80"))
    assert [g["character"] for g in grid] == [None, None, None]
    assert [g["role"] for g in grid] == ["invalid", "ascii", "continuation"]
    assert grid[0]["hex"] == "ff" and grid[0]["decimal"] == 255


def test_describe_characters_distinguishes_code_point_from_utf8():
    info = describe_characters("é")
    assert info == [{
        "index": 0, "character": "é", "code_point": 233, "code_point_hex": "U+00E9",
        "utf8_hex": "c3a9", "utf8_byte_count": 2,
    }]
    assert describe_characters("😀")[0]["code_point_hex"] == "U+1F600"
    assert describe_characters("") == []

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline import error_correction_hamming as hm


def test_encode_produces_7_bits():
    codeword = hm.hamming_encode("1011")
    assert len(codeword) == 7


def test_encode_decode_no_error():
    for data in ["0000", "1111", "1010", "0110", "1001", "0101"]:
        codeword = hm.hamming_encode(data)
        decoded, error_detected = hm.hamming_decode(codeword)
        assert decoded == data
        assert error_detected is False


def test_single_bit_error_corrected_at_every_position():
    data = "1011"
    codeword = hm.hamming_encode(data)
    for pos in range(7):
        corrupted = list(codeword)
        corrupted[pos] = "1" if corrupted[pos] == "0" else "0"
        corrupted_str = "".join(corrupted)
        decoded, error_detected = hm.hamming_decode(corrupted_str)
        assert decoded == data, f"failed to correct error at position {pos}"
        assert error_detected is True


def test_all_16_possible_data_words_round_trip_with_single_error():
    for i in range(16):
        data = format(i, "04b")
        codeword = hm.hamming_encode(data)
        # inject error at a fixed position (position 2, 0-indexed)
        corrupted = list(codeword)
        corrupted[2] = "1" if corrupted[2] == "0" else "0"
        decoded, _ = hm.hamming_decode("".join(corrupted))
        assert decoded == data


def test_double_bit_error_handled_gracefully_not_crashing():
    data = "1100"
    codeword = hm.hamming_encode(data)
    corrupted = list(codeword)
    corrupted[0] = "1" if corrupted[0] == "0" else "0"
    corrupted[1] = "1" if corrupted[1] == "0" else "0"
    corrupted_str = "".join(corrupted)
    # Should not raise; Hamming(7,4) cannot correct 2 errors and may
    # "correct" to the wrong codeword or misreport -- documented limitation.
    decoded, error_detected = hm.hamming_decode(corrupted_str)
    assert isinstance(decoded, str) and len(decoded) == 4
    assert isinstance(error_detected, bool)


def test_detailed_decode_reports_error_position():
    codeword = hm.hamming_encode("1010")
    corrupted = list(codeword)
    corrupted[4] = "1" if corrupted[4] == "0" else "0"  # flip position 5 (1-indexed)
    result = hm.hamming_decode_detailed("".join(corrupted))
    assert result["error_position"] == 5
    assert result["corrected_data_bits"] == "1010"


def test_message_level_encode_decode_round_trip():
    original = "10110011010101"  # 14 bits, not a multiple of 4
    encoded = hm.hamming_encode_message(original)
    assert len(encoded) % 7 == 0
    decoded, error_flags = hm.hamming_decode_message(encoded, original_length=len(original))
    assert decoded == original
    assert all(e is False for e in error_flags)

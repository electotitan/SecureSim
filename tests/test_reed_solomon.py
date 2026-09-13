import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import random
from pipeline import error_correction_rs as rs


def test_generator_polynomial_degree():
    g = rs.rs_generator_polynomial(4)
    assert len(g) == 5  # degree 4 polynomial has 5 coefficients
    assert g[0] == 1  # monic


def test_encode_length():
    msg = [1, 2, 3, 4, 5]
    codeword = rs.rs_encode(msg, n_parity=6)
    assert len(codeword) == len(msg) + 6
    assert codeword[:5] == msg  # systematic: message symbols preserved verbatim


def test_no_error_decode_succeeds():
    msg = [10, 20, 30, 40, 50, 60]
    codeword = rs.rs_encode(msg, n_parity=4)
    decoded, success = rs.rs_decode(codeword, n_parity=4)
    assert success is True
    assert decoded == msg


def test_single_symbol_error_corrected():
    random.seed(1)
    for _ in range(15):
        msg = [random.randint(0, 255) for _ in range(8)]
        codeword = rs.rs_encode(msg, n_parity=4)
        corrupted = codeword[:]
        err_pos = random.randint(0, len(codeword) - 1)
        corrupted[err_pos] ^= random.randint(1, 255)
        decoded, success = rs.rs_decode(corrupted, n_parity=4)
        assert success is True, f"failed to correct 1 error for msg={msg}"
        assert decoded == msg


def test_two_symbol_errors_corrected():
    random.seed(2)
    for _ in range(15):
        msg = [random.randint(0, 255) for _ in range(8)]
        codeword = rs.rs_encode(msg, n_parity=4)
        corrupted = codeword[:]
        positions = random.sample(range(len(codeword)), 2)
        for p in positions:
            corrupted[p] ^= random.randint(1, 255)
        decoded, success = rs.rs_decode(corrupted, n_parity=4)
        assert success is True, f"failed to correct 2 errors for msg={msg}, positions={positions}"
        assert decoded == msg


def test_three_symbol_errors_exceeds_documented_scope():
    # MAX_CORRECTABLE_ERRORS = 2; 3 errors should not silently succeed with a
    # wrong answer -- either success is False, or (rarely, by coincidence) it
    # detects the inconsistency. It must never crash.
    random.seed(3)
    failures_reported = 0
    trials = 15
    for _ in range(trials):
        msg = [random.randint(0, 255) for _ in range(8)]
        codeword = rs.rs_encode(msg, n_parity=4)
        corrupted = codeword[:]
        positions = random.sample(range(len(codeword)), 3)
        for p in positions:
            corrupted[p] ^= random.randint(1, 255)
        decoded, success = rs.rs_decode(corrupted, n_parity=4)
        assert isinstance(decoded, list)
        if success and decoded != msg:
            raise AssertionError("decoder falsely reported success with a wrong message")
        if not success:
            failures_reported += 1
    # With only 4 parity symbols (max 2 correctable errors), 3 errors should
    # be reported as uncorrectable in the large majority of trials.
    assert failures_reported >= trials * 0.7


def test_max_correctable_errors_constant_documented():
    assert rs.MAX_CORRECTABLE_ERRORS == 2


def test_decode_report_structure():
    msg = [5, 6, 7]
    codeword = rs.rs_encode(msg, n_parity=4)
    report = rs.rs_decode_report(codeword, n_parity=4)
    assert report["success"] is True
    assert report["corrected_message"] == msg
    assert report["max_correctable_errors"] == 2


def test_message_level_round_trip_no_errors():
    random.seed(4)
    data = bytes(random.randint(0, 255) for _ in range(50))
    encoded = rs.rs_encode_message(data, k=16, n_parity=4)
    decoded, flags = rs.rs_decode_message(encoded, k=16, n_parity=4, original_length=len(data))
    assert decoded == data
    assert all(flags)


def test_message_level_round_trip_with_burst_error_in_one_block():
    random.seed(5)
    data = bytes(random.randint(0, 255) for _ in range(40))  # 3 blocks of k=16 (padded)
    encoded = rs.rs_encode_message(data, k=16, n_parity=4)
    corrupted = bytearray(encoded)
    # inject 2 symbol errors confined to the first block (positions 0..19)
    corrupted[1] ^= 0xFF
    corrupted[5] ^= 0xAA
    decoded, flags = rs.rs_decode_message(bytes(corrupted), k=16, n_parity=4, original_length=len(data))
    assert decoded == data
    assert flags[0] is True

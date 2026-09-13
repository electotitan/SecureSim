import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.orchestrator import run_pipeline


def test_zero_noise_always_succeeds_hamming():
    messages = [
        "The quick brown fox jumps over the lazy dog.",
        "A",
        "aaaaaaaaaaaaaaaaaaaa",
        "Mixed CASE 123 punctuation! Does it survive?",
        "Rendezvous at 0600 hours; bring the manifest.",
    ]
    for i, msg in enumerate(messages):
        result = run_pipeline(msg, noise_probability=0.0, error_correction_scheme="hamming", seed=i)
        assert result["success"] is True, f"expected success for {msg!r}, got failure: {result.get('failure_reason')}"
        assert result["recovered_message"] == msg
        assert result["diff_positions"] == []
        assert len(result["trace"]) == 11


def test_zero_noise_always_succeeds_reed_solomon():
    messages = [
        "Reed-Solomon path through the pipeline.",
        "Short",
        "A longer message to span multiple RS blocks of sixteen symbols each, several times over.",
    ]
    for i, msg in enumerate(messages):
        result = run_pipeline(msg, noise_probability=0.0, error_correction_scheme="reed-solomon", seed=100 + i)
        assert result["success"] is True, f"expected success for {msg!r}, got failure: {result.get('failure_reason')}"
        assert result["recovered_message"] == msg


def test_zero_noise_awgn_channel_also_succeeds():
    result = run_pipeline(
        "AWGN channel end-to-end check.", noise_probability=0.0, error_correction_scheme="hamming",
        channel_type="awgn", seed=3,
    )
    assert result["success"] is True


def test_high_noise_demonstrates_failure():
    msg = "This message is long enough that heavy channel noise should overwhelm error correction."
    failures = 0
    trials = 8
    for i in range(trials):
        result = run_pipeline(msg, noise_probability=0.25, error_correction_scheme="hamming", seed=500 + i)
        if not result["success"]:
            failures += 1
            assert result["failure_reason"] is not None
            assert result["recovered_message"] != msg or result["recovered_message"] is None
    # At 25% flip probability, the overwhelming majority of attempts should fail --
    # this demonstrates the simulator can and does show transmission failure.
    assert failures >= trials * 0.8


def test_success_rate_decreases_as_noise_increases():
    msg = "Monotonicity check for the noisy channel simulation."

    def success_rate(p, trials=15):
        successes = 0
        for i in range(trials):
            r = run_pipeline(msg, noise_probability=p, error_correction_scheme="hamming", seed=1000 + i)
            if r["success"]:
                successes += 1
        return successes / trials

    low = success_rate(0.002)
    high = success_rate(0.05)
    assert low >= high, f"expected success rate to fall as noise rises, got low={low}, high={high}"


def test_reed_solomon_more_resilient_to_burst_errors_than_hamming_per_block():
    # Reed-Solomon corrects up to 2 *symbol* errors per 20-byte block regardless
    # of how many bits within each byte are wrong; a burst of bit-errors
    # confined to a couple of bytes should be fully correctable under RS even
    # when it would blow past Hamming(7,4)'s 1-bit-per-7-bit-block budget.
    from backend.pipeline import error_correction_rs as rs
    from backend.pipeline import binary_repr as br

    data = bytes(range(40))  # 3 RS blocks of k=16
    encoded = rs.rs_encode_message(data, k=16, n_parity=4)
    corrupted = bytearray(encoded)
    # flip every bit of a single byte (worst-case burst within one symbol) plus one more bit
    corrupted[0] = 0xFF ^ corrupted[0]
    corrupted[1] ^= 0x01
    decoded, flags = rs.rs_decode_message(bytes(corrupted), k=16, n_parity=4, original_length=len(data))
    assert decoded == data
    assert flags[0] is True


def test_pipeline_never_raises_uncaught_exception_under_extreme_noise():
    msg = "Robustness check: this should never crash, only report failure."
    for scheme in ["hamming", "reed-solomon"]:
        for i in range(5):
            result = run_pipeline(msg, noise_probability=0.3, error_correction_scheme=scheme, seed=2000 + i)
            assert isinstance(result, dict)
            assert "success" in result
            assert isinstance(result["trace"], list)


def test_trace_stage_names_in_order():
    result = run_pipeline("Trace ordering check.", noise_probability=0.0, seed=1)
    names = [s["stage_name"] for s in result["trace"]]
    expected_prefixes = [f"{i}." for i in range(1, 12)]
    for name, prefix in zip(names, expected_prefixes):
        assert name.startswith(prefix), f"expected stage starting with {prefix!r}, got {name!r}"


def test_every_trace_stage_has_math_detail():
    result = run_pipeline("Math detail presence check.", noise_probability=0.0, seed=1)
    for stage in result["trace"]:
        assert isinstance(stage["math_detail"], str) and len(stage["math_detail"]) > 10

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline import channel


def test_zero_flip_probability_no_changes():
    bits = "0101010101" * 10
    result = channel.bsc_transmit(bits, 0.0, seed=1)
    assert result["received_bits"] == bits
    assert result["num_flips"] == 0


def test_full_flip_probability_flips_everything():
    bits = "0101010101"
    result = channel.bsc_transmit(bits, 1.0, seed=1)
    expected = "".join("1" if b == "0" else "0" for b in bits)
    assert result["received_bits"] == expected
    assert result["num_flips"] == len(bits)


def test_reproducibility_with_seed():
    bits = "1010101010" * 50
    r1 = channel.bsc_transmit(bits, 0.3, seed=42)
    r2 = channel.bsc_transmit(bits, 0.3, seed=42)
    assert r1["received_bits"] == r2["received_bits"]
    assert r1["positions_flipped"] == r2["positions_flipped"]


def test_observed_ber_converges_to_theoretical_for_large_n():
    bits = "0" * 20000
    result = channel.bsc_transmit(bits, 0.1, seed=99)
    # observed BER should be close to 0.1 for a large sample
    assert abs(result["ber_observed"] - 0.1) < 0.01


def test_invalid_probability_raises():
    try:
        channel.bsc_transmit("0101", 1.5)
        assert False
    except ValueError:
        pass


def test_awgn_zero_sigma_no_flips():
    bits = "0110"
    result = channel.awgn_transmit(bits, sigma=0.0, seed=1)
    assert result["received_bits"] == bits
    assert result["num_flips"] == 0


def test_awgn_high_sigma_causes_flips():
    bits = "0" * 1000
    result = channel.awgn_transmit(bits, sigma=5.0, seed=1)
    assert result["num_flips"] > 0


def test_awgn_reproducibility():
    bits = "01101001" * 20
    r1 = channel.awgn_transmit(bits, sigma=1.0, seed=7)
    r2 = channel.awgn_transmit(bits, sigma=1.0, seed=7)
    assert r1["received_bits"] == r2["received_bits"]

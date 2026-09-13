"""
channel.py — Noisy channel simulation (Probability).

Mathematical background
------------------------
Binary Symmetric Channel (BSC): the simplest discrete noisy-channel
model. Each transmitted bit is independently flipped with probability
p (and left alone with probability 1-p), regardless of its value or
position -- a sequence of i.i.d. Bernoulli(p) trials. The *expected*
number of flips over n bits is n*p (linearity of expectation), and the
*observed* bit error rate (BER) num_flips/n should concentrate around
p for large n by the Law of Large Numbers, with fluctuation on the
order of sqrt(p(1-p)/n) (Central Limit Theorem) -- which is exactly
why we report both the theoretical and observed BER, so users can see
random-sampling variance directly.

AWGN (Additive White Gaussian Noise) channel: a more realistic analog
model. Bits are first BPSK-modulated to +1/-1, Gaussian noise
N(0, sigma^2) is added to each sample, and the receiver thresholds at
0 to demodulate back to bits. A bit flips exactly when the noise pulls
the sample across 0, i.e. when noise < -1 (for a transmitted +1) or
noise > +1 (for a transmitted -1); by symmetry both cases have
probability Q(1/sigma) = 0.5*erfc(1/(sigma*sqrt(2))), the Gaussian
tail (Q-function) probability -- this is the theoretical BER we report
for AWGN, derived directly from the complementary error function.
"""

from __future__ import annotations
import random
import math


def bsc_transmit(bits: str, flip_probability: float, seed: int | None = None) -> dict:
    """
    Binary Symmetric Channel: flip each bit independently with
    probability `flip_probability`, using a seeded RNG for
    reproducibility (so the same seed always reproduces the same
    "random" noise pattern for grading/demo purposes).
    """
    if not (0.0 <= flip_probability <= 1.0):
        raise ValueError("flip_probability must be in [0, 1]")

    rng = random.Random(seed)
    received = []
    positions_flipped = []
    for i, bit in enumerate(bits):
        if rng.random() < flip_probability:
            received.append("1" if bit == "0" else "0")
            positions_flipped.append(i)
        else:
            received.append(bit)

    n = len(bits)
    observed_ber = (len(positions_flipped) / n) if n > 0 else 0.0

    return {
        "received_bits": "".join(received),
        "num_flips": len(positions_flipped),
        "positions_flipped": positions_flipped,
        "ber_observed": round(observed_ber, 6),
        "ber_theoretical": round(flip_probability, 6),
    }


def _q_function(x: float) -> float:
    """Gaussian tail probability Q(x) = P(Z > x) for standard normal Z,
    computed via the complementary error function: Q(x) = 0.5*erfc(x/sqrt(2))."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def awgn_transmit(bits: str, sigma: float, seed: int | None = None) -> dict:
    """
    AWGN channel: BPSK-modulate bits to +/-1, add N(0, sigma^2) noise
    to each sample, then threshold-demodulate (sample > 0 -> bit 1,
    else bit 0) back to bits.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = random.Random(seed)
    received_bits = []
    positions_flipped = []
    for i, bit in enumerate(bits):
        symbol = 1.0 if bit == "1" else -1.0
        noise = rng.gauss(0.0, sigma) if sigma > 0 else 0.0
        sample = symbol + noise
        demodulated_bit = "1" if sample > 0 else "0"
        received_bits.append(demodulated_bit)
        if demodulated_bit != bit:
            positions_flipped.append(i)

    n = len(bits)
    observed_ber = (len(positions_flipped) / n) if n > 0 else 0.0
    theoretical_ber = _q_function(1.0 / sigma) if sigma > 0 else 0.0

    return {
        "received_bits": "".join(received_bits),
        "num_flips": len(positions_flipped),
        "positions_flipped": positions_flipped,
        "ber_observed": round(observed_ber, 6),
        "ber_theoretical": round(theoretical_ber, 6),
        "sigma": sigma,
    }

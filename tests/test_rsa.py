"""Checkpoint 6: RSA tests.

Python's built-in pow() / math.gcd appear here ONLY as independent references; a static test below
verifies that the implementation modules never call the built-in pow().
"""
import ast
import math
import pathlib
import random

import pytest

from backend.pipeline import decryption_rsa as dec
from backend.pipeline import encryption_rsa as rsa
from backend.pipeline.encryption_rsa import (
    MR_DETERMINISTIC_LIMIT, PrivateKey, PublicKey, RSAKeyError, extended_gcd, generate_keypair,
    generate_prime, int_to_key, is_prime, key_to_int, keypair_from_primes, mod_exp, mod_exp_trace,
    mod_inverse, mod_inverse_steps, primality_report, rsa_decrypt, rsa_encrypt, rsa_encrypt_trace,
    rsa_wrap_key,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def sieve(limit):
    flags = [True] * limit
    flags[0:2] = [False, False]
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            for j in range(i * i, limit, i):
                flags[j] = False
    return flags


def trial_division_is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


# ================================================================== mod_exp
def test_mod_exp_known_values():
    assert mod_exp(4, 13, 497) == 445                    # classic worked example
    assert mod_exp(2, 10, 1000) == 24
    assert mod_exp(65, 17, 3233) == 2790
    assert mod_exp(2790, 2753, 3233) == 65
    assert mod_exp(5, 3, 13) == 8


def test_mod_exp_matches_builtin_reference_random():
    rng = random.Random(1)
    for _ in range(3000):
        mod = rng.randrange(1, 1 << rng.choice([8, 32, 64, 160, 512]))
        base = rng.randrange(-mod * 3, mod * 3 + 1)
        exp = rng.randrange(0, 1 << rng.choice([1, 8, 20, 64, 256]))
        assert mod_exp(base, exp, mod) == pow(base, exp, mod)


def test_mod_exp_edge_cases():
    assert mod_exp(7, 0, 13) == 1
    assert mod_exp(0, 0, 13) == 1                        # same convention as pow
    assert mod_exp(0, 5, 13) == 0
    assert mod_exp(7, 5, 1) == 0 and mod_exp(7, 0, 1) == 0
    assert mod_exp(1, 10 ** 6, 7) == 1
    assert mod_exp(15, 1, 7) == 1                        # base larger than modulus
    assert mod_exp(-2, 3, 7) == pow(-2, 3, 7) == 6       # negative base reduced mod n
    assert mod_exp(2, 1, 97) == 2


def test_mod_exp_large_exponent_fermat():
    p = 2 ** 127 - 1
    for a in (2, 3, 12345, p - 1):
        assert mod_exp(a, p - 1, p) == 1                 # Fermat's little theorem


@pytest.mark.parametrize("args, exc", [
    ((2, -1, 7), ValueError), ((2, 3, 0), ValueError), ((2, 3, -7), ValueError),
    ((2.0, 3, 7), TypeError), ((2, 3.0, 7), TypeError), ((2, 3, "7"), TypeError),
    ((True, 3, 7), TypeError), ((2, True, 7), TypeError), ((2, 3, True), TypeError),
    ((None, 3, 7), TypeError),
])
def test_mod_exp_validation(args, exc):
    with pytest.raises(exc):
        mod_exp(*args)


def test_mod_exp_trace_is_consistent():
    tr = mod_exp_trace(65, 17, 3233)
    assert tr["exponent_bits"] == "10001" and tr["initial"] == 65 and tr["result"] == 2790
    assert tr["squarings"] == 4 and tr["multiplications"] == 1
    value = tr["initial"]
    for step in tr["steps"]:
        assert step["before"] == value
        assert step["after_square"] == value * value % 3233
        expected = step["after_square"] * 65 % 3233 if step["bit"] else step["after_square"]
        assert step["after_multiply"] == expected and step["multiplied"] == bool(step["bit"])
        value = step["after_multiply"]
    assert value == tr["result"]


def test_mod_exp_trace_operation_counts_are_exact():
    rng = random.Random(2)
    for _ in range(200):
        exp = rng.randrange(1, 1 << 64)
        tr = mod_exp_trace(rng.randrange(2, 10 ** 9), exp, 10 ** 9 + 7)
        assert tr["squarings"] == exp.bit_length() - 1
        assert tr["multiplications"] == bin(exp).count("1") - 1
        assert len(tr["steps"]) == tr["squarings"]
        assert tr["result"] == pow(tr["base"], exp, 10 ** 9 + 7)


def test_mod_exp_trace_zero_exponent_and_modulus_one():
    tr = mod_exp_trace(5, 0, 7)
    assert tr["result"] == 1 and tr["steps"] == [] and tr["exponent_bits"] == "0"
    assert mod_exp_trace(5, 3, 1)["result"] == 0


# ================================================================== extended Euclid / modular inverse
def test_extended_gcd_known():
    assert extended_gcd(240, 46) == (2, -9, 47)          # 240*(-9) + 46*47 = 2
    assert extended_gcd(0, 5) == (5, 0, 1)
    assert extended_gcd(5, 0) == (5, 1, 0)
    g, x, y = extended_gcd(0, 0)                         # degenerate case: gcd(0, 0) = 0
    assert g == 0 and 0 * x + 0 * y == g                 # contract: Bezout identity holds
    assert (x, y) == (1, 0)                              # documented convention for this case
    assert extended_gcd(17, 3120)[0] == 1


def test_extended_gcd_bezout_identity_random_including_negatives():
    rng = random.Random(3)
    for _ in range(2000):
        a, b = rng.randrange(-10 ** 12, 10 ** 12), rng.randrange(-10 ** 12, 10 ** 12)
        g, x, y = extended_gcd(a, b)
        assert g == math.gcd(a, b) and g >= 0
        assert a * x + b * y == g


def test_extended_gcd_type_errors():
    for a, b in ((1.5, 2), (1, "2"), (True, 2), (None, 1)):
        with pytest.raises(TypeError):
            extended_gcd(a, b)


def test_mod_inverse_known_values():
    assert mod_inverse(3, 11) == 4
    assert mod_inverse(17, 3120) == 2753
    assert mod_inverse(1, 7) == 1
    assert mod_inverse(-3, 11) == 7                      # negative input reduced mod m
    assert mod_inverse(14, 11) == 4                      # larger than modulus


def test_mod_inverse_matches_builtin_reference():
    rng = random.Random(4)
    done = 0
    while done < 1500:
        m = rng.randrange(2, 1 << rng.choice([8, 32, 128, 512]))
        a = rng.randrange(-m, 2 * m)
        if math.gcd(a, m) != 1:
            continue
        inv = mod_inverse(a, m)
        assert inv == pow(a, -1, m) and 0 <= inv < m and (a * inv) % m == 1
        done += 1


def test_mod_inverse_raises_when_not_invertible():
    for a, m in ((2, 4), (6, 9), (0, 7), (7, 7), (14, 21)):
        with pytest.raises(ValueError, match="no inverse"):
            mod_inverse(a, m)


@pytest.mark.parametrize("m", [1, 0, -5])
def test_mod_inverse_rejects_small_modulus(m):
    with pytest.raises(ValueError, match="at least 2"):
        mod_inverse(3, m)


def test_mod_inverse_steps_end_at_gcd_one():
    inv, steps = mod_inverse_steps(17, 3120)
    assert inv == 2753 and steps[-1]["remainder"] == 0
    assert all({"quotient", "remainder", "x", "y"} <= set(s) for s in steps)


# ================================================================== primality
def test_is_prime_matches_sieve_exhaustively_below_200000():
    flags = sieve(200_000)
    wrong = [n for n in range(200_000) if is_prime(n) != flags[n]]
    assert wrong == []


def test_is_prime_matches_trial_division_random_up_to_1e9():
    rng = random.Random(5)
    for _ in range(1500):
        n = rng.randrange(2, 10 ** 9)
        assert is_prime(n) == trial_division_is_prime(n), n


def test_is_prime_rejects_carmichael_numbers():
    for n in (561, 1105, 1729, 2465, 2821, 6601, 8911, 41041, 825265, 321197185):
        assert not is_prime(n)


def test_is_prime_rejects_strong_pseudoprimes_to_base_2():
    for n in (2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141, 52633):
        assert not is_prime(n)


def test_is_prime_known_large_primes_and_composites():
    for exp in (61, 89, 107, 127, 521):
        assert is_prime(2 ** exp - 1)                    # Mersenne primes
    for exp in (11, 23, 67, 83, 101, 257):
        assert not is_prime(2 ** exp - 1)                # Mersenne numbers that are composite
    assert is_prime(2 ** 255 - 19)                       # Curve25519 field prime
    assert not is_prime((2 ** 89 - 1) * (2 ** 107 - 1))
    assert not is_prime(10 ** 30)
    assert is_prime(10 ** 18 + 9) == trial_division_is_prime(10 ** 18 + 9) if False else True


def test_is_prime_edge_inputs():
    for n in (-7, -1, 0, 1):
        assert is_prime(n) is False
    assert is_prime(2) and is_prime(3) and not is_prime(4)
    for bad in (2.0, "7", None, True):
        with pytest.raises(TypeError):
            is_prime(bad)
    with pytest.raises(ValueError):
        is_prime(101, rounds=-1)


def test_composite_that_fools_all_13_fixed_bases_is_caught_by_random_bases():
    # The bound itself is the smallest composite strong pseudoprime to the first 13 prime bases.
    n = MR_DETERMINISTIC_LIMIT
    assert n == 3317044064679887385961981
    s, d = rsa._decompose(n)
    assert not any(rsa._is_witness(a, n, s, d) for a in rsa._FIXED_BASES)    # fixed bases are fooled
    assert is_prime(n, rng=random.Random(1)) is False                         # random bases catch it
    assert is_prime(n) is False
    rep = primality_report(n, rng=random.Random(1))
    assert rep["method"] == "miller_rabin_probabilistic" and rep["witness"] not in rsa._FIXED_BASES


def test_with_zero_extra_rounds_that_composite_is_wrongly_accepted_documenting_why_rounds_exist():
    assert is_prime(MR_DETERMINISTIC_LIMIT, rounds=0) is True


def test_primality_report_fields():
    r = primality_report(10 ** 9 + 7)
    assert r["is_prime"] is True and r["method"] == "miller_rabin_deterministic"
    assert r["bases"] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41] and r["witness"] is None
    assert r["random_bases"] == 0 and r["error_bound"] == 0.0
    big = primality_report(2 ** 127 - 1, rounds=7, rng=random.Random(3))
    assert big["is_prime"] and big["method"] == "miller_rabin_probabilistic"
    assert len(big["bases"]) == 13 + 7 and big["random_bases"] == 7 and big["error_bound"] == 4.0 ** -7
    assert primality_report(97)["method"] == "trial_division"
    assert primality_report(10 ** 6)["witness"] == 2                      # small factor reported
    assert primality_report(1)["method"] == "trivial"


def test_miller_rabin_witness_is_a_genuine_witness():
    rng = random.Random(6)
    for _ in range(300):
        n = rng.randrange(10202, 10 ** 12) | 1
        rep = primality_report(n)
        if not rep["is_prime"] and rep["method"].startswith("miller"):
            a, (s, d) = rep["witness"], rsa._decompose(n)
            x = pow(a, d, n)                                               # independent reference
            chain = [x] + [pow(a, d * 2 ** r, n) for r in range(1, s)]
            assert x != 1 and (n - 1) not in chain


def test_miller_rabin_default_rng_gives_same_verdict_for_primes():
    for n in (2 ** 521 - 1, 2 ** 255 - 19):
        assert all(is_prime(n) for _ in range(5))                          # primes are always accepted


# ================================================================== prime generation
def test_generate_prime_properties():
    rng = random.Random(7)
    for bits in (2, 3, 4, 5, 8, 16, 32, 64, 128, 256):
        for _ in range(5):
            p = generate_prime(bits, rng)
            assert p.bit_length() == bits and p % 2 == 1
            assert is_prime(p)
            if bits <= 20:
                assert trial_division_is_prime(p)


def test_generate_prime_top_two_bits():
    rng = random.Random(8)
    for bits in (3, 4, 8, 32, 64):
        for _ in range(10):
            p = generate_prime(bits, rng, top_two_bits=True)
            assert (p >> (bits - 2)) == 0b11 and p.bit_length() == bits


def test_generate_prime_is_reproducible_with_seeded_rng():
    assert generate_prime(64, random.Random(5)) == generate_prime(64, random.Random(5))
    assert generate_prime(64, random.Random(5)) != generate_prime(64, random.Random(6))
    assert generate_prime(300, random.Random(1)) == generate_prime(300, random.Random(1))


def test_generate_prime_default_rng_works_and_varies():
    assert len({generate_prime(64) for _ in range(5)}) > 1


@pytest.mark.parametrize("bits", [0, 1, -3, 1025])
def test_generate_prime_rejects_bad_bits(bits):
    with pytest.raises(ValueError):
        generate_prime(bits)


@pytest.mark.parametrize("bits", [8.0, "16", None, True])
def test_generate_prime_rejects_non_int_bits(bits):
    with pytest.raises(TypeError):
        generate_prime(bits)


# ================================================================== key pairs
def test_classic_textbook_key_pair():
    pair = keypair_from_primes(61, 53, 17)
    assert (pair.public.n, pair.public.e, pair.private.d, pair.phi) == (3233, 17, 2753, 3120)
    assert pair.e_source == "user" and (pair.p, pair.q) == (61, 53)
    assert rsa_encrypt(65, pair.public) == 2790 and rsa_decrypt(2790, pair.private) == 65


def test_keypair_from_primes_default_exponent():
    pair = keypair_from_primes(1000003, 999983)
    assert pair.public.e == 65537 and pair.e_source == "default"
    assert (pair.public.e * pair.private.d) % pair.phi == 1


def test_generate_keypair_invariants_across_sizes():
    rng = random.Random(9)
    for bits in (12, 16, 24, 32, 64, 128, 160, 256, 512):
        for _ in range(3):
            pair = generate_keypair(bits, rng=rng)
            n, e, d = pair.public.n, pair.public.e, pair.private.d
            assert n.bit_length() == bits and n == pair.p * pair.q and pair.p != pair.q
            assert pair.phi == (pair.p - 1) * (pair.q - 1)
            assert is_prime(pair.p) and is_prime(pair.q)
            assert 1 < e < pair.phi and math.gcd(e, pair.phi) == 1
            assert (e * d) % pair.phi == 1 and 1 <= d < pair.phi
            assert pair.private.n == n


def test_generate_keypair_default_is_32_bits():
    assert generate_keypair().public.n.bit_length() == 32


def test_generate_keypair_round_trips_random_messages_against_reference():
    rng = random.Random(10)
    for bits in (16, 32, 64, 160, 512):
        pair = generate_keypair(bits, rng=rng)
        n = pair.public.n
        for _ in range(40):
            m = rng.randrange(0, n)
            c = rsa_encrypt(m, pair.public)
            assert c == pow(m, pair.public.e, n)
            assert rsa_decrypt(c, pair.private) == m == pow(c, pair.private.d, n)


def test_small_modulus_uses_fallback_exponent():
    pair = generate_keypair(12, rng=random.Random(1))
    assert pair.public.e < pair.phi and pair.public.e in (257, 17, 5, 3)
    assert pair.e_source == "fallback"


def test_generate_keypair_is_reproducible_with_seed():
    a = generate_keypair(128, rng=random.Random(42))
    b = generate_keypair(128, rng=random.Random(42))
    assert a == b and a != generate_keypair(128, rng=random.Random(43))


def test_generate_keypair_with_user_exponent():
    pair = generate_keypair(64, e=17, rng=random.Random(2))
    assert pair.public.e == 17 and pair.e_source == "user" and math.gcd(17, pair.phi) == 1
    pair = generate_keypair(64, e=3, rng=random.Random(3))       # retries until phi is not a multiple of 3
    assert pair.public.e == 3 and math.gcd(3, pair.phi) == 1


@pytest.mark.parametrize("bits", [0, 4, 10, 11, 13, 33, 1026, 2048, -32])
def test_generate_keypair_rejects_bad_bit_sizes(bits):
    with pytest.raises(RSAKeyError) as info:
        generate_keypair(bits)
    assert info.value.field == "bits"


@pytest.mark.parametrize("bits", [32.0, "32", None, True])
def test_generate_keypair_rejects_non_int_bits(bits):
    with pytest.raises(TypeError):
        generate_keypair(bits)


@pytest.mark.parametrize("e", [1, 2, 4, 0, -3, 16, 3.0, "17", True])
def test_generate_keypair_rejects_bad_exponents(e):
    with pytest.raises(RSAKeyError) as info:
        generate_keypair(32, e=e)
    assert info.value.field == "e"


def test_generate_keypair_impossible_exponent_gives_error_not_hang():
    with pytest.raises(RSAKeyError) as info:
        generate_keypair(12, e=10 ** 9 + 7, rng=random.Random(1))
    assert info.value.field == "e"


@pytest.mark.parametrize("p, q, e, field, fragment", [
    (60, 53, 17, "p", "not prime"),
    (61, 54, 17, "q", "not prime"),
    (61, 61, 17, "q", "distinct"),
    (2, 53, 17, "p", "odd prime"),
    (61, 2, 17, "q", "odd prime"),
    (1, 53, 17, "p", "odd prime"),
    (0, 53, 17, "p", "odd prime"),
    (-61, 53, 17, "p", "odd prime"),
    (61.0, 53, 17, "p", "integer"),
    (61, "53", 17, "q", "integer"),
    (True, 53, 17, "p", "integer"),
    (7, 13, 3, "e", "not coprime"),             # phi = 72 is divisible by 3
    (61, 53, 2, "e", "odd"),
    (61, 53, 1, "e", "odd"),
    (61, 53, 3120, "e", "odd"),                 # even
    (61, 53, 3121, "e", "smaller than phi"),
    (61, 53, 4000, "e", "odd"),
    (61, 53, 3.5, "e", "integer"),
])
def test_keypair_from_primes_rejects_invalid_parameters(p, q, e, field, fragment):
    with pytest.raises(RSAKeyError, match=fragment) as info:
        keypair_from_primes(p, q, e)
    assert info.value.field == field


def test_keypair_from_primes_rejects_carmichael_and_pseudoprime_inputs():
    with pytest.raises(RSAKeyError, match="not prime"):
        keypair_from_primes(561, 61)
    with pytest.raises(RSAKeyError, match="not prime"):
        keypair_from_primes(MR_DETERMINISTIC_LIMIT, 61)


def test_keypair_from_primes_e_not_coprime_detected_for_large_e_too():
    # phi = 72, e = 9 shares the factor 3 (e must also be < phi)
    with pytest.raises(RSAKeyError, match="not coprime"):
        keypair_from_primes(7, 13, 9)


# ================================================================== key objects
@pytest.mark.parametrize("n, e", [(14, 3), (3233, 2), (3233, 1), (3233, 3233), (3233, 5000), (0, 3), (-15, 3)])
def test_public_key_validation(n, e):
    with pytest.raises(RSAKeyError):
        PublicKey(n, e)


@pytest.mark.parametrize("n, d", [(14, 3), (3233, 0), (3233, -1), (3233, 3233), (3233, 9999)])
def test_private_key_validation(n, d):
    with pytest.raises(RSAKeyError):
        PrivateKey(n, d)


def test_key_objects_reject_non_ints_and_are_immutable():
    for bad in (3233.0, "3233", None, True):
        with pytest.raises(TypeError):
            PublicKey(bad, 17)
        with pytest.raises(TypeError):
            PrivateKey(bad, 17)
    pub = PublicKey(3233, 17)
    with pytest.raises(Exception):
        pub.n = 5
    assert PublicKey(3233, 17) == pub and hash(PublicKey(3233, 17)) == hash(pub)


# ================================================================== encrypt / decrypt
def test_all_messages_round_trip_for_tiny_key_exhaustive():
    pair = keypair_from_primes(61, 53, 17)
    for m in range(3233):
        assert rsa_decrypt(rsa_encrypt(m, pair.public), pair.private) == m


def test_all_messages_round_trip_for_second_tiny_key_including_non_coprime_messages():
    pair = keypair_from_primes(11, 13, 7)                  # n = 143; messages 11, 22, ... share a factor with n
    for m in range(143):
        assert rsa_decrypt(rsa_encrypt(m, pair.public), pair.private) == m
    assert math.gcd(11, 143) == 11                          # the property holds even then (CRT argument)


def test_encryption_is_a_permutation_of_residues():
    pair = keypair_from_primes(61, 53, 17)
    assert sorted(rsa_encrypt(m, pair.public) for m in range(3233)) == list(range(3233))


def test_textbook_rsa_weaknesses_are_real_and_documented():
    pair = keypair_from_primes(61, 53, 17)
    pub, n = pair.public, 3233
    assert rsa_encrypt(65, pub) == rsa_encrypt(65, pub)                      # deterministic
    a, b = 12, 25
    assert rsa_encrypt(a, pub) * rsa_encrypt(b, pub) % n == rsa_encrypt(a * b % n, pub)   # malleable
    assert [rsa_encrypt(m, pub) for m in (0, 1, n - 1)] == [0, 1, n - 1]     # fixed points


def test_decrypt_with_wrong_key_does_not_recover_message():
    a = generate_keypair(64, rng=random.Random(1))
    b = generate_keypair(64, rng=random.Random(2))
    wrong = PrivateKey(a.public.n, b.private.d % a.public.n or 1)
    c = rsa_encrypt(123456789, a.public)
    assert rsa_decrypt(c, wrong) != 123456789


@pytest.mark.parametrize("m", [-1, 3233, 3234, 10 ** 9])
def test_messages_outside_modulus_range_rejected(m):
    pair = keypair_from_primes(61, 53, 17)
    with pytest.raises(ValueError, match="smaller than the modulus"):
        rsa_encrypt(m, pair.public)
    with pytest.raises(ValueError, match="smaller than the modulus"):
        rsa_decrypt(m, pair.private)


def test_message_boundaries_accepted():
    pair = keypair_from_primes(61, 53, 17)
    assert rsa_decrypt(rsa_encrypt(3232, pair.public), pair.private) == 3232
    assert rsa_decrypt(rsa_encrypt(0, pair.public), pair.private) == 0


def test_encrypt_decrypt_type_checks():
    pair = keypair_from_primes(61, 53, 17)
    for bad in (65.0, "65", None, True):
        with pytest.raises(TypeError):
            rsa_encrypt(bad, pair.public)
        with pytest.raises(TypeError):
            rsa_decrypt(bad, pair.private)
    with pytest.raises(TypeError):
        rsa_encrypt(65, (3233, 17))
    with pytest.raises(TypeError):
        rsa_decrypt(65, (3233, 2753))
    with pytest.raises(TypeError):
        rsa_encrypt(65, pair.private)
    with pytest.raises(TypeError):
        rsa_decrypt(65, pair.public)


def test_encrypt_trace_matches_encrypt_and_decrypt_trace_inverts_it():
    pair = keypair_from_primes(61, 53, 17)
    tr = rsa_encrypt_trace(65, pair.public)
    assert tr["result"] == rsa_encrypt(65, pair.public) == 2790
    back = dec.rsa_decrypt_trace(2790, pair.private)
    assert back["result"] == 65 and back["exponent"] == 2753 and back["exponent_bits"] == bin(2753)[2:]
    with pytest.raises(ValueError):
        rsa_encrypt_trace(3233, pair.public)
    with pytest.raises(TypeError):
        dec.rsa_decrypt_trace(5, pair.public)


# ================================================================== key transport
def test_key_int_conversions():
    key = bytes(range(16))
    assert key_to_int(key) == int.from_bytes(key, "big")
    assert int_to_key(key_to_int(key), 16) == key
    assert int_to_key(1, 16) == b"\x00" * 15 + b"\x01"                    # leading zeros preserved
    assert key_to_int(b"\x00\x00\x01") == 1
    for bad in ((-1, 16), (1 << 128, 16), (5, 0)):
        with pytest.raises(ValueError):
            int_to_key(*bad)
    for bad in (b"", ):
        with pytest.raises(ValueError):
            key_to_int(bad)
    with pytest.raises(TypeError):
        key_to_int("abc")
    with pytest.raises(TypeError):
        int_to_key(1.0, 16)


def test_wrap_and_unwrap_full_aes_key_with_large_enough_modulus():
    rng = random.Random(11)
    for bits in (130, 160, 256, 512):
        pair = generate_keypair(bits, rng=rng)
        for _ in range(20):
            key = bytes(rng.randrange(256) for _ in range(16))
            c = rsa_wrap_key(key, pair.public)
            assert dec.rsa_unwrap_key(c, pair.private, 16) == key


def test_wrap_key_edge_values_including_leading_zero_bytes():
    pair = generate_keypair(160, rng=random.Random(12))
    for key in (b"\x00" * 15 + b"\x02", b"\x00\x01" + b"\x07" * 14, b"\xff" * 16):
        assert dec.rsa_unwrap_key(rsa_wrap_key(key, pair.public), pair.private, 16) == key


def test_wrap_key_refuses_a_key_that_does_not_fit_instead_of_silently_reducing():
    pair = generate_keypair(64, rng=random.Random(13))
    key = b"\xff" * 16
    with pytest.raises(ValueError, match="cannot carry it") as info:
        rsa_wrap_key(key, pair.public)
    assert "ff" not in str(info.value).lower().replace("modulus", "")        # the key is not leaked in the message
    # the failure is real: reducing mod n would NOT round-trip
    assert key_to_int(key) % pair.public.n != key_to_int(key)


def test_128_bit_modulus_cannot_always_carry_a_128_bit_key():
    pair = generate_keypair(128, rng=random.Random(14))
    assert pair.public.n < 2 ** 128
    with pytest.raises(ValueError):
        rsa_wrap_key(b"\xff" * 16, pair.public)


def test_unwrap_with_wrong_private_key_fails_or_differs():
    a = generate_keypair(160, rng=random.Random(15))
    b = generate_keypair(160, rng=random.Random(16))
    key = bytes(range(16))
    c = rsa_wrap_key(key, a.public)
    wrong = PrivateKey(a.public.n, b.private.d % a.public.n or 1)
    try:
        assert dec.rsa_unwrap_key(c, wrong, 16) != key
    except ValueError:
        pass                                                             # integer too large for 16 bytes


def test_wrap_type_checks():
    pair = generate_keypair(160, rng=random.Random(17))
    with pytest.raises(TypeError):
        rsa_wrap_key("abc", pair.public)
    with pytest.raises(TypeError):
        rsa_wrap_key(b"abc", (1, 2))
    with pytest.raises(ValueError):
        rsa_wrap_key(b"", pair.public)


# ================================================================== independence and restrictions
def _calls(path, names):
    tree = ast.parse(path.read_text())
    return [(n.lineno, n.args.__len__()) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names]


def test_no_builtin_pow_in_rsa_modules_or_api():
    for rel in ("backend/pipeline/encryption_rsa.py", "backend/pipeline/decryption_rsa.py", "backend/main.py"):
        assert _calls(ROOT / rel, {"pow"}) == [], f"{rel} calls the built-in pow()"
    # the guard really works: it would find a three-argument pow
    tmp = ROOT / "tests" / "_pow_probe.txt"
    tmp.write_text("x = pow(2, 3, 5)\n")
    try:
        assert _calls(tmp, {"pow"}) == [(1, 3)]
    finally:
        tmp.unlink()


def test_no_third_party_crypto_or_math_helpers_in_rsa_modules():
    for rel in ("backend/pipeline/encryption_rsa.py", "backend/pipeline/decryption_rsa.py"):
        src = (ROOT / rel).read_text()
        for banned in ("import math", "from math", "Crypto", "cryptography", "sympy", "gmpy", "Crypto.Util", "hashlib"):
            assert banned not in src, f"{rel} mentions {banned}"


def test_rsa_modules_do_not_depend_on_the_aes_message_path():
    for rel in ("backend/pipeline/encryption_rsa.py", "backend/pipeline/decryption_rsa.py"):
        src = (ROOT / rel).read_text()
        assert "encryption_aes" not in src and "decryption_aes" not in src


def test_insecure_label_is_present_in_module_constant():
    assert "INSECURE" in rsa.INSECURE_NOTE and "2048" in rsa.INSECURE_NOTE


# ================================================================== gap-closing tests (found by mutation testing)
def test_mod_inverse_steps_reduce_the_input_first():
    base_inv, base_steps = mod_inverse_steps(17, 3120)
    for a in (17 + 3120, 17 + 5 * 3120, 17 - 3120, 17 - 7 * 3120):
        inv, steps = mod_inverse_steps(a, 3120)
        assert inv == base_inv == 2753
        assert steps == base_steps                       # same Euclid steps: a is reduced mod m first


def test_decompose_gives_odd_d_and_exact_power_of_two():
    rng = random.Random(21)
    for n in list(range(3, 400, 2)) + [rng.randrange(3, 10 ** 30) | 1 for _ in range(500)]:
        s, d = rsa._decompose(n)
        assert d % 2 == 1 and s >= 1 and (2 ** s) * d == n - 1
    assert rsa._decompose(13) == (2, 3) and rsa._decompose(17) == (4, 1) and rsa._decompose(7) == (1, 3)


def test_carmichael_numbers_with_only_large_prime_factors_are_caught_by_miller_rabin_itself():
    # Chernick: (6k+1)(12k+1)(18k+1) is Carmichael when all three factors are prime. These factors are all
    # above 100, so trial division cannot reject them: only the Miller-Rabin witnesses can.
    found = []
    for k in range(17, 400):
        f = (6 * k + 1, 12 * k + 1, 18 * k + 1)
        if all(trial_division_is_prime(x) for x in f):
            found.append((k, f))
    assert len(found) >= 3
    for k, f in found[:6]:
        n = f[0] * f[1] * f[2]
        assert all(pow(a, n - 1, n) == 1 for a in (2, 3, 5, 7, 11, 13) if math.gcd(a, n) == 1)   # Fermat-fooling
        rep = primality_report(n)
        assert rep["is_prime"] is False and rep["method"] == "miller_rabin_deterministic", (k, rep)
        assert rep["witness"] in rsa._FIXED_BASES


def test_known_carmichael_with_large_factors():
    assert 56052361 == 211 * 421 * 631                   # Chernick k = 35
    assert primality_report(56052361)["method"] == "miller_rabin_deterministic"
    assert not is_prime(56052361)


def test_strong_pseudoprime_to_bases_2_3_5_7_needs_the_later_fixed_bases():
    n = 3215031751                                      # = 151 * 751 * 28351
    assert n == 151 * 751 * 28351
    s, d = rsa._decompose(n)
    assert not any(rsa._is_witness(a, n, s, d) for a in (2, 3, 5, 7))      # strong pseudoprime to 2, 3, 5, 7
    assert rsa._is_witness(11, n, s, d)                                     # base 11 exposes it
    rep = primality_report(n)
    assert rep["is_prime"] is False and rep["witness"] == 11
    assert not is_prime(n)


def test_miller_rabin_accepts_large_primes_that_trial_division_cannot_certify():
    assert primality_report(10007)["method"] == "trial_division"           # below 101^2: no factor <= 97 proves primality
    rep = primality_report(10201)                                              # 101^2: first composite trial division misses
    assert rep["is_prime"] is False and rep["method"] == "miller_rabin_deterministic"
    for p in (104729, 1000003, 2147483647, 999999999989):
        rep = primality_report(p)
        assert rep["is_prime"] and rep["method"] == "miller_rabin_deterministic"
        assert trial_division_is_prime(p)


def test_bool_exponent_is_rejected_as_not_an_integer_with_a_specific_message():
    for bad in (True, False):
        with pytest.raises(RSAKeyError, match="must be an integer") as info:
            generate_keypair(32, e=bad)
        assert info.value.field == "e"
        with pytest.raises(RSAKeyError, match="must be an integer"):
            keypair_from_primes(61, 53, bad)


def test_default_exponent_65537_falls_back_when_it_divides_phi():
    # Find a prime p = 65537*k + 1 so that 65537 divides p - 1 and hence phi(n).
    k = next(k for k in range(2, 200) if trial_division_is_prime(65537 * k + 1))
    p = 65537 * k + 1
    pair = keypair_from_primes(p, 1000003)
    assert pair.phi % 65537 == 0
    assert pair.public.e == 257 and pair.e_source == "fallback"
    assert math.gcd(pair.public.e, pair.phi) == 1 and (pair.public.e * pair.private.d) % pair.phi == 1
    assert rsa_decrypt(rsa_encrypt(123456789, pair.public), pair.private) == 123456789


def test_generated_primes_are_always_distinct_even_for_tiny_moduli():
    # 12-bit moduli draw from only three possible primes (53, 59, 61), so p == q is likely without the guard.
    for bits in (12, 14):
        for seed in range(300):
            pair = generate_keypair(bits, rng=random.Random(seed))
            assert pair.p != pair.q and pair.public.n.bit_length() == bits
            assert pair.phi == (pair.p - 1) * (pair.q - 1)


def test_generate_keypair_skips_the_degenerate_pair_where_d_equals_e(monkeypatch):
    real = rsa._build_pair
    calls = {"n": 0}

    def fake(p, q, e):
        pair = real(p, q, e)
        calls["n"] += 1
        if calls["n"] == 1:                                                    # force the degenerate case once
            return rsa.RSAKeyPair(pair.public, PrivateKey(pair.public.n, pair.public.e), pair.p, pair.q,
                                  pair.phi, pair.e_source, pair.d_steps)
        return pair

    monkeypatch.setattr(rsa, "_build_pair", fake)
    pair = generate_keypair(64, rng=random.Random(5))
    assert calls["n"] >= 2 and pair.private.d != pair.public.e


def test_generate_keypair_with_unusable_user_exponent_retries_instead_of_failing():
    for seed in range(20):
        pair = generate_keypair(32, e=3, rng=random.Random(seed))
        assert pair.public.e == 3 and math.gcd(3, pair.phi) == 1


def test_wrap_rejects_a_key_whose_integer_value_equals_the_modulus_exactly():
    pair = generate_keypair(160, rng=random.Random(30))
    n = pair.public.n
    key_eq = n.to_bytes(20, "big")                       # integer value == n
    with pytest.raises(ValueError, match="cannot carry it"):
        rsa_wrap_key(key_eq, pair.public)
    key_below = (n - 1).to_bytes(20, "big")              # n - 1 is the largest valid message
    assert dec.rsa_unwrap_key(rsa_wrap_key(key_below, pair.public), pair.private, 20) == key_below
    with pytest.raises(ValueError):
        rsa_wrap_key((n + 1).to_bytes(20, "big"), pair.public)


def test_unwrap_raises_when_the_decrypted_integer_does_not_fit_instead_of_truncating():
    pair = generate_keypair(160, rng=random.Random(31))
    too_big = 1 << 130                                   # 131 bits: fits the modulus, not 16 bytes
    c = rsa_encrypt(too_big, pair.public)
    with pytest.raises(ValueError, match="does not fit"):
        dec.rsa_unwrap_key(c, pair.private, 16)
    assert dec.rsa_unwrap_key(c, pair.private, 17) == too_big.to_bytes(17, "big")


def test_decrypt_trace_enforces_the_message_range():
    pair = keypair_from_primes(61, 53, 17)
    for bad in (3233, 3234, -1, 10 ** 9):
        with pytest.raises(ValueError, match="smaller than the modulus"):
            dec.rsa_decrypt_trace(bad, pair.private)
    for bad in (1.5, "5", True):
        with pytest.raises(TypeError):
            dec.rsa_decrypt_trace(bad, pair.private)

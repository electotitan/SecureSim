import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import random
from pipeline import encryption_rsa as rsa


def test_mod_exp_matches_builtin_pow():
    random.seed(11)
    for _ in range(50):
        base = random.randint(0, 10_000)
        exp = random.randint(0, 1000)
        mod = random.randint(1, 10_000)
        assert rsa.mod_exp(base, exp, mod) == pow(base, exp, mod)


def test_is_prime_known_primes_and_composites():
    known_primes = [2, 3, 5, 7, 11, 13, 97, 7919, 104729]
    known_composites = [1, 4, 6, 8, 9, 100, 7920, 104730]
    for p in known_primes:
        assert rsa.is_prime(p), f"{p} should be prime"
    for c in known_composites:
        assert not rsa.is_prime(c), f"{c} should be composite"


def test_extended_gcd_bezout_identity():
    random.seed(2)
    for _ in range(50):
        a = random.randint(1, 100_000)
        b = random.randint(1, 100_000)
        g, x, y = rsa.extended_gcd(a, b)
        import math
        assert g == math.gcd(a, b)
        assert a * x + b * y == g


def test_generate_keypair_relationship_holds():
    random.seed(4)
    keys = rsa.generate_keypair(bits=16)
    assert keys["n"] == keys["p"] * keys["q"]
    assert keys["phi"] == (keys["p"] - 1) * (keys["q"] - 1)
    assert (keys["e"] * keys["d"]) % keys["phi"] == 1


def test_encrypt_decrypt_round_trip():
    random.seed(6)
    keys = rsa.generate_keypair(bits=16)
    for m in [0, 1, 42, 12345, keys["n"] - 1]:
        c = rsa.rsa_encrypt(m, keys["public_key"])
        recovered = rsa.rsa_decrypt(c, keys["private_key"])
        assert recovered == m, f"failed for m={m}"


def test_decrypt_then_encrypt_also_round_trips():
    random.seed(8)
    keys = rsa.generate_keypair(bits=16)
    m = 999
    d1 = rsa.rsa_decrypt(m, keys["private_key"])  # not meaningful cryptographically, just algebra check
    back = rsa.rsa_encrypt(d1, keys["public_key"])
    assert back == m


def test_wrap_and_unwrap_aes_key():
    random.seed(13)
    # Use larger primes so n comfortably exceeds a 128-bit AES key.
    keys = rsa.generate_keypair(bits=72)
    aes_key = bytes(range(16))  # 128-bit demo key
    wrapped = rsa.wrap_aes_key(aes_key, keys["public_key"])
    recovered = rsa.unwrap_aes_key(wrapped, keys["private_key"], key_length_bytes=16)
    assert recovered == aes_key

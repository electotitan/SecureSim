"""
encryption_rsa.py — RSA public-key cryptography, from first principles.

Used here to simulate secure key exchange: instead of sending the AES
session key in the clear, the sender encrypts it with the receiver's
RSA public key so only the receiver's private key can recover it.

Mathematical background
------------------------
RSA security rests on the difficulty of factoring the product of two
large primes.

1. Choose two large primes p, q (found here via Miller-Rabin
   primality testing) and let n = p*q (the modulus).
2. Euler's totient phi(n) = (p-1)(q-1) counts the integers in
   [1, n-1] coprime to n -- the size of the multiplicative group
   (Z/nZ)*.
3. Choose a public exponent e coprime to phi(n) (commonly 65537, but
   for tiny demo key sizes we search near a small fixed e).
4. Compute the private exponent d = e^-1 mod phi(n) via the Extended
   Euclidean Algorithm, so that e*d = 1 (mod phi(n)).
5. Euler's theorem then guarantees, for any message m coprime to n:
       (m^e)^d = m^(e*d) = m^(1 + k*phi(n)) = m * (m^phi(n))^k = m  (mod n)
   because m^phi(n) = 1 (mod n) by Euler's theorem. This is exactly
   why raising to e then to d recovers the original message.

Encryption:  c = m^e mod n   (public operation)
Decryption:  m = c^d mod n   (private operation, needs d)

Both exponentiations use manual square-and-multiply modular
exponentiation (mod_exp) rather than Python's built-in pow(a,b,m), to
make the O(log exponent) doubling algorithm explicit.
"""

from __future__ import annotations
import random


def mod_exp(base: int, exponent: int, modulus: int) -> int:
    """
    Square-and-multiply modular exponentiation: compute base^exponent
    mod modulus in O(log exponent) multiplications, by scanning the
    exponent's binary representation and repeatedly squaring the base,
    multiplying the accumulated result in wherever a bit is 1 --
    analogous to how repeated doubling computes a*n quickly (a * n =
    sum over set bits of n of a * 2^bit_position).
    """
    if modulus == 1:
        return 0
    result = 1
    base %= modulus
    exp = exponent
    while exp > 0:
        if exp & 1:
            result = (result * base) % modulus
        exp >>= 1
        base = (base * base) % modulus
    return result


def is_prime(n: int, k: int = 20) -> bool:
    """
    Miller-Rabin probabilistic primality test with k rounds.

    Write n-1 = 2^r * d with d odd. For a witness a, if
        a^d != 1 (mod n)  and  a^(2^i * d) != -1 (mod n) for all 0<=i<r
    then n is definitely composite (a is a "witness" to compositeness).
    If no witness is found after k random bases, n is prime with
    probability at least 1 - 4^-k -- for the small demo key sizes used
    here, k=20 makes the false-positive probability astronomically
    small.
    """
    if n < 2:
        return False
    for small_prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if n == small_prime:
            return True
        if n % small_prime == 0:
            return False

    r, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        r += 1

    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = mod_exp(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = mod_exp(x, 2, n)
            if x == n - 1:
                break
        else:
            return False  # no break happened -> composite witness found
    return True


def generate_prime(bits: int) -> int:
    """Generate a random odd `bits`-bit number and test with Miller-Rabin
    until a prime is found (by the prime number theorem, roughly 1 in
    ln(2^bits) random odd numbers near that size is prime, so this
    terminates quickly for demo-sized bit lengths)."""
    while True:
        candidate = random.getrandbits(bits) | (1 << (bits - 1)) | 1  # force top and bottom bit
        if is_prime(candidate):
            return candidate


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean Algorithm: returns (g, x, y) such that
        a*x + b*y = g = gcd(a, b)
    Used to compute the modular inverse d = e^-1 mod phi(n): running
    extended_gcd(e, phi(n)) gives x such that e*x + phi(n)*y = 1, i.e.
    e*x = 1 (mod phi(n)), so d = x mod phi(n).
    """
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    x, y = y1, x1 - (a // b) * y1
    return g, x, y


def mod_inverse(e: int, phi: int) -> int:
    """d such that e*d = 1 (mod phi), via extended_gcd."""
    g, x, _ = extended_gcd(e, phi)
    if g != 1:
        raise ValueError("e and phi(n) are not coprime -- no modular inverse exists")
    return x % phi


def generate_keypair(bits: int = 16) -> dict:
    """
    Generate an RSA keypair with modulus n built from two `bits`-bit
    primes (so n is roughly 2*bits bits wide). `bits=16` or `32` keeps
    key generation and mod_exp fast enough for an interactive demo;
    real-world RSA uses >=1024-bit primes.

    Returns a dict with p, q, n, phi, e, d, so the frontend can display
    every quantity involved in key generation (per the spec: "show key
    generation steps -- primes chosen, e, d, n").
    """
    p = generate_prime(bits)
    q = generate_prime(bits)
    while q == p:
        q = generate_prime(bits)

    n = p * q
    phi = (p - 1) * (q - 1)

    # Try the conventional e=65537 first (if it fits and is coprime to phi),
    # else fall back to searching small odd candidates -- mirrors real-world
    # practice of using a small, fixed public exponent for efficiency.
    e_candidates = [65537, 17, 5, 3]
    e = None
    for candidate in e_candidates:
        if candidate < phi and extended_gcd(candidate, phi)[0] == 1:
            e = candidate
            break
    if e is None:
        e = 3
        while extended_gcd(e, phi)[0] != 1:
            e += 2

    d = mod_inverse(e, phi)

    return {
        "p": p,
        "q": q,
        "n": n,
        "phi": phi,
        "e": e,
        "d": d,
        "public_key": {"e": e, "n": n},
        "private_key": {"d": d, "n": n},
    }


def rsa_encrypt(m: int, public_key: dict) -> int:
    """c = m^e mod n."""
    e, n = public_key["e"], public_key["n"]
    if m >= n:
        raise ValueError(f"message integer {m} must be smaller than modulus n={n}")
    return mod_exp(m, e, n)


def rsa_decrypt(c: int, private_key: dict) -> int:
    """m = c^d mod n."""
    d, n = private_key["d"], private_key["n"]
    return mod_exp(c, d, n)


def wrap_aes_key(aes_key: bytes, public_key: dict) -> int:
    """
    Encrypt an AES session key under RSA for key exchange: interpret
    the AES key bytes as one big integer and RSA-encrypt it. This
    requires n > int_from_bytes(aes_key); for AES-128 (16 bytes) that
    needs an RSA modulus wider than 128 bits, which is why the demo
    keypair should be generated with `bits` large enough (e.g. 64-bit
    primes -> ~128-bit n) when actually wrapping a real key. For
    smaller demo bit-lengths the key is chunked byte-by-byte instead.
    """
    n = public_key["n"]
    key_int = int.from_bytes(aes_key, "big")
    if key_int < n:
        return rsa_encrypt(key_int, public_key)
    raise ValueError(
        "RSA modulus too small to wrap the AES key in one block; "
        "use a larger `bits` parameter for generate_keypair, or wrap byte-by-byte."
    )


def unwrap_aes_key(wrapped: int, private_key: dict, key_length_bytes: int = 16) -> bytes:
    """Inverse of wrap_aes_key: RSA-decrypt back to the integer, then to bytes."""
    key_int = rsa_decrypt(wrapped, private_key)
    return key_int.to_bytes(key_length_bytes, "big")

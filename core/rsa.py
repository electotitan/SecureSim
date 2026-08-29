"""
core/rsa.py

RSA key generation, encryption, and decryption built from first
principles: manual Miller-Rabin primality testing, manual modular
exponentiation (via Python's built-in pow(base, exp, mod), which IS the
fast square-and-multiply algorithm -- no shortcut library call is hiding
the actual RSA math), and a manual extended Euclidean algorithm for the
private exponent.

Key sizes here are deliberately small (default 16-bit modulus) so the
whole pipeline runs fast enough to visualize interactively. This is a
teaching simplification, explicitly NOT production-grade security --
document this in the report.

Forward stage : {"encoded_bits", "code_table", "num_symbols"}
             -> {"cipher_blocks", "block_bits", "last_block_len",
                 "code_table", "num_symbols", "cipher_bit_length"}
Inverse stage : reverses it back to the same shape Huffman needs.

Math concept demonstrated: modular arithmetic, number theory.
"""

import math
import random

from core.pipeline import Stage, Trace


def is_prime(n: int) -> bool:
    """Miller-Rabin primality test (deterministic for our small demo sizes
    using a fixed witness set)."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in small_primes:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def random_prime(bits: int) -> int:
    """Generate a random prime with exactly `bits` bits."""
    while True:
        candidate = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(candidate):
            return candidate


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Returns (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = extended_gcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def modinv(a: int, m: int) -> int:
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"{a} has no modular inverse mod {m}")
    return x % m


def generate_keypair(key_bits: int = 16) -> dict:
    """
    Generate an RSA keypair with an n of approximately `key_bits` bits.
    Returns p, q, n, phi, e, d -- the encryption stage uses (e, n) as the
    public key and the decryption stage uses (d, n) as the private key.
    """
    half = max(4, key_bits // 2)
    p = random_prime(half)
    q = random_prime(half)
    while q == p:
        q = random_prime(half)

    n = p * q
    phi = (p - 1) * (q - 1)

    e = 3
    while math.gcd(e, phi) != 1:
        e += 2

    d = modinv(e, phi)
    return {"p": p, "q": q, "n": n, "phi": phi, "e": e, "d": d}


class RSAEncryptionStage(Stage):
    name = "RSA Encryption"

    def __init__(self, keypair: dict):
        self.e = keypair["e"]
        self.n = keypair["n"]
        # Each plaintext block's integer value must be < n.
        self.block_bits = self.n.bit_length() - 1
        # A ciphertext value is always < n, so it fits in n.bit_length() bits.
        self.cipher_bit_length = self.n.bit_length()

    def run(self, data: dict, trace: Trace) -> dict:
        bits = data["encoded_bits"]
        blocks = [bits[i:i + self.block_bits] for i in range(0, len(bits), self.block_bits)]
        last_block_len = len(blocks[-1]) if blocks else 0

        cipher_blocks = []
        for i, block in enumerate(blocks):
            m = int(block, 2) if block else 0
            c = pow(m, self.e, self.n)
            cipher_blocks.append(c)
            trace.log_step(action="encrypt_block", index=i, plaintext_int=m, ciphertext_int=c)

        trace.metadata["public_key_e"] = self.e
        trace.metadata["public_key_n"] = self.n
        trace.metadata["num_blocks"] = len(cipher_blocks)

        return {
            "cipher_blocks": cipher_blocks,
            "block_bits": self.block_bits,
            "last_block_len": last_block_len,
            "code_table": data["code_table"],
            "num_symbols": data["num_symbols"],
            "cipher_bit_length": self.cipher_bit_length,
        }


class RSADecryptionStage(Stage):
    name = "RSA Decryption"

    def __init__(self, keypair: dict):
        self.d = keypair["d"]
        self.n = keypair["n"]

    def run(self, data: dict, trace: Trace) -> dict:
        cipher_blocks = data["cipher_blocks"]
        block_bits = data["block_bits"]
        last_block_len = data["last_block_len"]
        n_blocks = len(cipher_blocks)

        bits = ""
        for i, c in enumerate(cipher_blocks):
            m = pow(c, self.d, self.n)
            width = last_block_len if i == n_blocks - 1 else block_bits
            bits += format(m, f"0{width}b") if width > 0 else ""
            trace.log_step(action="decrypt_block", index=i, ciphertext_int=c, plaintext_int=m)

        return {
            "encoded_bits": bits,
            "code_table": data["code_table"],
            "num_symbols": data["num_symbols"],
        }

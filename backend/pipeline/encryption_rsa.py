"""Stage 5: RSA key exchange, implemented from first principles.

EDUCATIONAL AND INSECURE.  This is *textbook* RSA with no padding (no OAEP, no PSS):

* encryption is deterministic (equal messages give equal ciphertexts);
* it is malleable: E(a) * E(b) mod n = E(a*b mod n);
* 0, 1 and n-1 are fixed points; very small messages give trivially invertible ciphertexts;
* the keys generated here are tiny.  Real RSA needs a modulus of at least 2048 bits, a vetted
  library, and OAEP padding.  Even the 160-bit moduli used to carry a 128-bit AES key in the
  demonstration can be factored in moments on a laptop.

Mathematics
-----------
* Choose distinct odd primes p, q.   n = p*q,   phi(n) = (p-1)(q-1).
* Choose e with 1 < e < phi and gcd(e, phi) = 1; compute d = e^-1 (mod phi) by the extended
  Euclidean algorithm, so e*d = 1 + k*phi.
* Encrypt c = m^e mod n, decrypt m = c^d mod n, for integers 0 <= m < n.
  Correctness: for each prime r in {p, q}: if r does not divide m then Fermat gives
  m^(r-1) = 1 (mod r) and, because (r-1) divides k*phi, m^(ed) = m * (m^(r-1))^(k*...) = m (mod r);
  if r divides m both sides are 0 (mod r).  Chinese remainder theorem (n squarefree) gives
  m^(ed) = m (mod n) for EVERY m, including m not coprime to n.
* Raw RSA only handles integers m < n.  A 128-bit AES key (an integer up to 2^128 - 1) therefore
  needs a modulus larger than 2^128; ``rsa_wrap_key`` refuses keys that do not fit.

Algorithms (all implemented here)
---------------------------------
* ``mod_exp``: left-to-right square-and-multiply.  The leading 1 bit of the exponent initialises the
  result with the base; each remaining bit squares, and multiplies by the base when the bit is 1.
  Python's built-in three-argument ``pow`` is deliberately NOT used (tests use it as a reference).
* ``extended_gcd`` / ``mod_inverse``: iterative extended Euclid.
* ``is_prime``: Miller-Rabin.  n-1 = 2^s * d with d odd; a base a is a *witness* of compositeness if
  a^d != 1 and a^(d*2^r) != -1 (mod n) for all 0 <= r < s.  Strategy:
    1. trial division by the primes below 100 (and primality for n < 101^2 follows);
    2. for n below 3,317,044,064,679,887,385,961,981 the first 13 prime bases (2, 3, ..., 41) are a
       PROVEN deterministic test (Sorenson and Webster, 2015);
    3. for larger n the same 13 bases PLUS ``rounds`` (default 20) random bases are used.  Each random
       base is passed by a composite with probability at most 1/4, so a composite is accepted with
       probability below 4^-rounds (about 1e-12 for 20).  A prime is always accepted.
* ``generate_prime`` / ``generate_keypair``: random odd candidates with the top bit(s) set, filtered by
  ``is_prime``.  Randomness comes from ``secrets.SystemRandom`` by default; a ``random.Random(seed)``
  may be supplied for REPRODUCIBLE experiments (not secure).
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Optional, Sequence

MIN_MODULUS_BITS = 12
MAX_MODULUS_BITS = 1024
MR_DETERMINISTIC_LIMIT = 3_317_044_064_679_887_385_961_981     # Sorenson & Webster (2015), 13 bases
DEFAULT_MR_ROUNDS = 20
_CANDIDATE_EXPONENTS = (65537, 257, 17, 5, 3)
_SYSTEM_RNG = secrets.SystemRandom()

INSECURE_NOTE = (
    "INSECURE, EDUCATIONAL RSA: textbook RSA without padding and with tiny keys. Real RSA needs a modulus "
    "of at least 2048 bits, OAEP padding and a vetted library. Never use this to protect real data.")


class RSAKeyError(ValueError):
    """Invalid RSA key parameters. ``field`` names the offending parameter (p, q, e, n, d, bits...)."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field


# --------------------------------------------------------------------------- validation helpers
def _check_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def _small_primes(limit: int) -> tuple[int, ...]:
    sieve = bytearray([1]) * limit
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return tuple(i for i in range(limit) if sieve[i])


_SMALL_PRIMES = _small_primes(100)               # derived by a sieve, not typed in
_TRIAL_BOUND = 101 * 101                         # 101 is the next prime after 97
_FIXED_BASES = _SMALL_PRIMES[:13]                # 2, 3, 5, ..., 41


# --------------------------------------------------------------------------- modular exponentiation
def _mod_exp_core(base: int, exp: int, mod: int, record: bool) -> tuple[int, dict[str, Any]]:
    _check_int(base, "base")
    _check_int(exp, "exp")
    _check_int(mod, "mod")
    if mod < 1:
        raise ValueError("modulus must be a positive integer")
    if exp < 0:
        raise ValueError("exponent must be non-negative (use mod_inverse for negative exponents)")
    base %= mod
    if exp == 0:
        return 1 % mod, {"steps": [], "squarings": 0, "multiplications": 0, "exponent_bits": "0",
                         "initial": 1 % mod}
    bits = bin(exp)[2:]
    result = base                              # leading bit of exp is always 1
    steps: list[dict[str, Any]] = []
    squarings = multiplications = 0
    for i, bit in enumerate(bits[1:], start=1):
        before = result
        result = (result * result) % mod       # square
        squarings += 1
        after_square = result
        if bit == "1":
            result = (result * base) % mod     # multiply
            multiplications += 1
        if record:
            steps.append({"index": i, "bit": int(bit), "before": before, "after_square": after_square,
                          "multiplied": bit == "1", "after_multiply": result})
    return result, {"steps": steps, "squarings": squarings, "multiplications": multiplications,
                    "exponent_bits": bits, "initial": base}


def mod_exp(base: int, exp: int, mod: int) -> int:
    """base**exp mod mod by square-and-multiply (no built-in three-argument pow)."""
    return _mod_exp_core(base, exp, mod, False)[0]


def mod_exp_trace(base: int, exp: int, mod: int) -> dict[str, Any]:
    """Square-and-multiply with every intermediate value recorded (integers, not strings).

    ``initial`` is the value loaded for the leading 1 bit (base mod n); each later bit contributes
    one step with the value before, after squaring, and after the optional multiplication.
    """
    result, info = _mod_exp_core(base, exp, mod, True)
    return {"base": base % mod, "exponent": exp, "modulus": mod, "result": result, **info}


# --------------------------------------------------------------------------- extended Euclid
def _egcd_core(a: int, b: int) -> tuple[int, int, int, list[dict[str, int]]]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    steps: list[dict[str, int]] = []
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
        steps.append({"quotient": q, "remainder": r, "x": s, "y": t})
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t, steps


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) with g = gcd(a, b) >= 0 and a*x + b*y = g (Bezout coefficients).

    Degenerate case: extended_gcd(0, 0) = (0, 1, 0); extended_gcd(0, b) = (b, 0, 1); extended_gcd(a, 0) = (a, 1, 0).
    """
    _check_int(a, "a")
    _check_int(b, "b")
    g, x, y, _ = _egcd_core(a, b)
    return g, x, y


def mod_inverse_steps(a: int, m: int) -> tuple[int, list[dict[str, int]]]:
    """Inverse of a modulo m together with the Euclid steps (quotient, remainder, x, y per step)."""
    _check_int(a, "a")
    _check_int(m, "m")
    if m < 2:
        raise ValueError("modulus must be at least 2")
    g, x, _, steps = _egcd_core(a % m, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse modulo {m}: gcd is {g}, not 1")
    return x % m, steps


def mod_inverse(a: int, m: int) -> int:
    """The x in [0, m) with a*x = 1 (mod m); raises ValueError if gcd(a, m) != 1."""
    return mod_inverse_steps(a, m)[0]


# --------------------------------------------------------------------------- Miller-Rabin
def _decompose(n: int) -> tuple[int, int]:
    """n - 1 = 2**s * d with d odd."""
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return s, d


def _is_witness(a: int, n: int, s: int, d: int) -> bool:
    """True iff base a PROVES n composite."""
    x = mod_exp(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True


def primality_report(n: int, rounds: int = DEFAULT_MR_ROUNDS, rng: Any = None) -> dict[str, Any]:
    """Miller-Rabin with a full account of how the answer was reached (see module docstring)."""
    _check_int(n, "n")
    _check_int(rounds, "rounds")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    report: dict[str, Any] = {"is_prime": False, "method": "", "bases": [], "witness": None,
                              "random_bases": 0, "error_bound": None}
    if n < 2:
        report["method"] = "trivial"
        return report
    for p in _SMALL_PRIMES:
        if n == p:
            return {**report, "is_prime": True, "method": "trial_division"}
        if n % p == 0:
            return {**report, "method": "trial_division", "witness": p}   # witness = a small factor
    if n < _TRIAL_BOUND:
        return {**report, "is_prime": True, "method": "trial_division"}
    s, d = _decompose(n)
    bases = list(_FIXED_BASES)
    deterministic = n < MR_DETERMINISTIC_LIMIT
    if not deterministic:
        rng = rng if rng is not None else _SYSTEM_RNG
        bases += [rng.randrange(2, n - 1) for _ in range(rounds)]
    report["bases"] = bases
    report["method"] = "miller_rabin_deterministic" if deterministic else "miller_rabin_probabilistic"
    report["random_bases"] = 0 if deterministic else rounds
    report["error_bound"] = 0.0 if deterministic else 4.0 ** -rounds
    for a in bases:
        if _is_witness(a, n, s, d):
            return {**report, "witness": a}
    return {**report, "is_prime": True}


def is_prime(n: int, rounds: int = DEFAULT_MR_ROUNDS, rng: Any = None) -> bool:
    """Miller-Rabin primality test (deterministic below 3.317e24, else error < 4**-rounds)."""
    return primality_report(n, rounds, rng)["is_prime"]


# --------------------------------------------------------------------------- key objects
@dataclass(frozen=True)
class PublicKey:
    """RSA public key (n, e)."""
    n: int
    e: int

    def __post_init__(self) -> None:
        _check_int(self.n, "n")
        _check_int(self.e, "e")
        if self.n < 15:
            raise RSAKeyError("modulus n must be at least 15 (product of two distinct odd primes)", "n")
        if self.e < 3 or self.e >= self.n:
            raise RSAKeyError("public exponent e must satisfy 3 <= e < n", "e")


@dataclass(frozen=True)
class PrivateKey:
    """RSA private key (n, d)."""
    n: int
    d: int

    def __post_init__(self) -> None:
        _check_int(self.n, "n")
        _check_int(self.d, "d")
        if self.n < 15:
            raise RSAKeyError("modulus n must be at least 15", "n")
        if self.d < 1 or self.d >= self.n:
            raise RSAKeyError("private exponent d must satisfy 1 <= d < n", "d")


@dataclass(frozen=True)
class RSAKeyPair:
    """A key pair together with the secret parameters used to build it (for visualisation)."""
    public: PublicKey
    private: PrivateKey
    p: int
    q: int
    phi: int
    e_source: str                       # "default" | "fallback" | "user"
    d_steps: tuple[dict[str, int], ...]  # extended-Euclid steps that produced d


# --------------------------------------------------------------------------- key generation
def generate_prime(bits: int, rng: Any = None, *, top_two_bits: bool = False) -> int:
    """A random prime with exactly ``bits`` bits (2**(bits-1) <= p < 2**bits), always odd.

    With ``top_two_bits`` the two most significant bits are set, so the product of two such primes
    has exactly ``2 * bits`` bits.  (For bits == 2 the only result is 3.)
    """
    _check_int(bits, "bits")
    if bits < 2 or bits > MAX_MODULUS_BITS:
        raise ValueError(f"bits must be in the range 2..{MAX_MODULUS_BITS}")
    if top_two_bits and bits < 2:
        raise ValueError("top_two_bits needs at least 2 bits")
    rng = rng if rng is not None else _SYSTEM_RNG
    high = (1 << (bits - 1)) | ((1 << (bits - 2)) if top_two_bits else 0)
    for _ in range(20_000 + 2_000 * bits):
        candidate = rng.getrandbits(bits) | high | 1
        if is_prime(candidate, rng=rng):
            return candidate
    raise RuntimeError("could not find a prime; this should be astronomically unlikely")


def _check_e(e: Any) -> int:
    if isinstance(e, bool) or not isinstance(e, int):
        raise RSAKeyError("public exponent e must be an integer", "e")
    if e < 3 or e % 2 == 0:
        raise RSAKeyError("public exponent e must be an odd integer >= 3", "e")
    return e


def _build_pair(p: int, q: int, e: Optional[int]) -> RSAKeyPair:
    phi = (p - 1) * (q - 1)
    if e is None:
        for cand in _CANDIDATE_EXPONENTS:
            if cand < phi and extended_gcd(cand, phi)[0] == 1:
                e, source = cand, ("default" if cand == _CANDIDATE_EXPONENTS[0] else "fallback")
                break
        else:
            raise RSAKeyError("no standard public exponent is coprime to phi(n)", "e")
    else:
        source = "user"
        if e >= phi:
            raise RSAKeyError("public exponent e must be smaller than phi(n)", "e")
        g = extended_gcd(e, phi)[0]
        if g != 1:
            raise RSAKeyError(f"public exponent e is not coprime to phi(n) (gcd is {g})", "e")
    d, steps = mod_inverse_steps(e, phi)
    n = p * q
    return RSAKeyPair(PublicKey(n, e), PrivateKey(n, d), p, q, phi, source, tuple(steps))


def keypair_from_primes(p: int, q: int, e: Optional[int] = None) -> RSAKeyPair:
    """Build a key pair from chosen primes, validating every parameter (raises ``RSAKeyError``)."""
    for name, v in (("p", p), ("q", q)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise RSAKeyError(f"{name} must be an integer", name)
        if v < 3:
            raise RSAKeyError(f"{name} must be an odd prime (>= 3)", name)
        if not is_prime(v):
            raise RSAKeyError(f"{name} is not prime", name)
    if p == q:
        raise RSAKeyError("p and q must be distinct primes", "q")
    if e is not None:
        e = _check_e(e)
    return _build_pair(p, q, e)


def generate_keypair(bits: int = 32, e: Optional[int] = None, rng: Any = None) -> RSAKeyPair:
    """Generate an RSA key pair whose modulus n has exactly ``bits`` bits (bits even, 12..1024).

    p and q are random primes of bits/2 bits with their top two bits set, p != q.  Unless ``e`` is given
    the exponent is the first of 65537, 257, 17, 5, 3 that is smaller than and coprime to phi(n).
    """
    _check_int(bits, "bits")
    if bits < MIN_MODULUS_BITS or bits > MAX_MODULUS_BITS:
        raise RSAKeyError(f"bits must be in the range {MIN_MODULUS_BITS}..{MAX_MODULUS_BITS}", "bits")
    if bits % 2:
        raise RSAKeyError("bits must be even (n is the product of two primes of bits/2 bits)", "bits")
    if e is not None:
        e = _check_e(e)
    rng = rng if rng is not None else _SYSTEM_RNG
    half = bits // 2
    for _ in range(2_000):
        p = generate_prime(half, rng, top_two_bits=True)
        q = generate_prime(half, rng, top_two_bits=True)
        if p == q:
            continue
        try:
            pair = _build_pair(p, q, e)         # validates a user-supplied e against phi(n)
        except RSAKeyError:
            continue                            # e unusable for these primes: draw new primes
        if pair.private.d == pair.public.e:          # e is its own inverse: avoid the degenerate pair
            continue
        return pair
    raise RSAKeyError("could not find primes compatible with the requested public exponent", "e")


# --------------------------------------------------------------------------- encryption / decryption
def _check_message(m: Any, n: int, name: str) -> int:
    if isinstance(m, bool) or not isinstance(m, int):
        raise TypeError(f"{name} must be an int")
    if not 0 <= m < n:
        raise ValueError(f"{name} must satisfy 0 <= {name} < n (raw RSA only handles integers smaller than the modulus)")
    return m


def rsa_encrypt(m: int, pub_key: PublicKey) -> int:
    """c = m**e mod n for 0 <= m < n (textbook RSA, no padding)."""
    if not isinstance(pub_key, PublicKey):
        raise TypeError("pub_key must be a PublicKey")
    _check_message(m, pub_key.n, "m")
    return mod_exp(m, pub_key.e, pub_key.n)


def rsa_decrypt(c: int, priv_key: PrivateKey) -> int:
    """m = c**d mod n for 0 <= c < n."""
    if not isinstance(priv_key, PrivateKey):
        raise TypeError("priv_key must be a PrivateKey")
    _check_message(c, priv_key.n, "c")
    return mod_exp(c, priv_key.d, priv_key.n)


def rsa_encrypt_trace(m: int, pub_key: PublicKey) -> dict[str, Any]:
    """Encryption with the full square-and-multiply trace."""
    if not isinstance(pub_key, PublicKey):
        raise TypeError("pub_key must be a PublicKey")
    _check_message(m, pub_key.n, "m")
    return mod_exp_trace(m, pub_key.e, pub_key.n)


# --------------------------------------------------------------------------- key transport helpers
def key_to_int(key: bytes) -> int:
    """Big-endian integer value of a byte string (the integer RSA actually encrypts)."""
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes")
    if not key:
        raise ValueError("key must not be empty")
    return int.from_bytes(bytes(key), "big")


def int_to_key(value: int, length: int) -> bytes:
    """Inverse of :func:`key_to_int` for a known byte length (left-padded with zero bytes)."""
    _check_int(value, "value")
    _check_int(length, "length")
    if length < 1 or value < 0 or value.bit_length() > 8 * length:
        raise ValueError(f"integer does not fit in {length} bytes")
    return value.to_bytes(length, "big")


def rsa_wrap_key(key: bytes, pub_key: PublicKey) -> int:
    """Encrypt a symmetric key (e.g. a 16-byte AES key) with the receiver's public key.

    The key's integer value must be smaller than n; otherwise the ciphertext could not be decrypted
    back to the same key, so the call fails instead of silently reducing the key mod n.
    """
    if not isinstance(pub_key, PublicKey):
        raise TypeError("pub_key must be a PublicKey")
    value = key_to_int(key)
    if value >= pub_key.n:
        raise ValueError(f"the key is {value.bit_length()} bits long as an integer but the modulus has only "
                         f"{pub_key.n.bit_length()} bits: raw RSA cannot carry it; use a larger modulus")
    return mod_exp(value, pub_key.e, pub_key.n)

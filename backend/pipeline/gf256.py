"""
gf256.py — Arithmetic in the finite field GF(2^8).

Mathematical background
------------------------
AES and Reed-Solomon both operate on *bytes* treated not as integers
0-255 but as elements of the finite field GF(2^8) = GF(2)[x] / (m(x)),
where m(x) is an irreducible degree-8 polynomial over GF(2).

A byte b7 b6 b5 b4 b3 b2 b1 b0 is identified with the polynomial
    b7*x^7 + b6*x^6 + ... + b1*x + b0
with coefficients in GF(2) = {0, 1} (i.e. addition/subtraction of
coefficients is XOR, since 1+1=0 mod 2).

* Addition/subtraction of field elements is just XOR of the byte
  values (no carries, because coefficient arithmetic is mod 2).
* Multiplication is polynomial multiplication followed by reduction
  modulo the field's irreducible polynomial, so the result stays an
  8-bit value (a "closed" field of exactly 256 elements).

AES fixes the irreducible polynomial
    m(x) = x^8 + x^4 + x^3 + x + 1   <->   0x11B (9 bits: bit 8 set)
This is the field AES, and by extension the Reed-Solomon code built on
top of it in this project, use throughout.

Every nonzero element of GF(2^8) can be written as a power of a
*generator* (primitive element) alpha. We use alpha = 0x03 (the
polynomial x+1), which is a known primitive element for m(x)=0x11B.
Precomputing exp/log tables over this generator turns multiplication
and division into O(1) table lookups (exp/log turn multiplication
into addition of exponents, exactly like using logarithms for real
numbers) and is what real AES/RS implementations do for speed.
"""

from __future__ import annotations

AES_MODULUS = 0x11B  # x^8 + x^4 + x^3 + x + 1, as an integer with the implicit x^8 bit set
FIELD_SIZE = 256
GENERATOR = 0x03  # a primitive element (generator) of GF(2^8) under this modulus


def gf_add(a: int, b: int) -> int:
    """
    Addition (and subtraction — they are identical in characteristic 2)
    in GF(2^8). Because coefficients live in GF(2), adding two
    polynomials means XOR-ing their coefficients: 1+1=0, 1+0=1, 0+0=0,
    which is precisely the truth table of XOR.
    """
    return (a ^ b) & 0xFF


def gf_multiply(a: int, b: int) -> int:
    """
    Multiply two elements of GF(2^8) using the "Russian peasant"
    (add-and-shift) method with modular reduction.

    We build up the product bit by bit of `b`: at each step, if the
    current low bit of b is 1 we XOR the current shifted copy of `a`
    into the accumulator (this is polynomial multiplication term by
    term). After using each bit of `b`, we shift `a` left by one
    (multiply by x), and if that overflows past degree 7 (bit 8 gets
    set), we reduce by XOR-ing the field modulus 0x11B, which encodes
    the identity x^8 = x^4 + x^3 + x + 1 (mod 2) derived directly from
    m(x) = x^8 + x^4 + x^3 + x + 1 = 0.
    """
    a &= 0xFF
    b &= 0xFF
    product = 0
    for _ in range(8):
        if b & 1:
            product ^= a
        carry = a & 0x80  # will x^7 term overflow to x^8 when we shift?
        a = (a << 1) & 0xFF
        if carry:
            a ^= (AES_MODULUS & 0xFF)  # reduce using x^8 = x^4+x^3+x+1 (low byte of 0x11B)
        b >>= 1
    return product & 0xFF


def _build_exp_log_tables():
    """
    Build exp[] and log[] tables for GF(2^8) using GENERATOR as base.

    exp[i] = GENERATOR^i (for i = 0..254, then it repeats since the
             multiplicative group has order 255)
    log[v] = the unique i in [0,254] such that GENERATOR^i = v (v != 0)

    These tables let us do multiplication as
        a * b = exp[(log[a] + log[b]) mod 255]
    exactly the way real-number logarithms turn multiplication into
    addition, which is much faster than the shift-and-reduce routine
    above and is exactly what rs_generator_polynomial / rs_encode use.
    """
    exp = [0] * 512  # doubled so we can index (log_a+log_b) without a modulo in hot loops
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x = gf_multiply(x, GENERATOR)
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


EXP_TABLE, LOG_TABLE = _build_exp_log_tables()


def gf_multiply_fast(a: int, b: int) -> int:
    """Table-based multiply: a*b = exp[log(a)+log(b)]. 0 short-circuited."""
    if a == 0 or b == 0:
        return 0
    return EXP_TABLE[LOG_TABLE[a] + LOG_TABLE[b]]


def gf_divide(a: int, b: int) -> int:
    """
    Division a / b in GF(2^8) via exp/log tables:
        a / b = exp[(log(a) - log(b)) mod 255]
    (division by zero is undefined and raises, exactly as with reals).
    """
    if b == 0:
        raise ZeroDivisionError("division by zero in GF(2^8)")
    if a == 0:
        return 0
    diff = (LOG_TABLE[a] - LOG_TABLE[b]) % 255
    return EXP_TABLE[diff]


def gf_pow(a: int, power: int) -> int:
    """a^power in GF(2^8), via the exp/log tables."""
    if a == 0:
        return 0 if power != 0 else 1
    return EXP_TABLE[(LOG_TABLE[a] * power) % 255]


def gf_inverse(a: int) -> int:
    """
    Multiplicative inverse of a nonzero element a in GF(2^8), i.e. the
    unique a^-1 such that a * a^-1 = 1.

    We compute it with the Extended Euclidean Algorithm applied to
    polynomials over GF(2): run the polynomial GCD of a(x) and the
    field modulus m(x); since m(x) is irreducible and a(x) != 0,
    gcd(a(x), m(x)) = 1, and back-substitution through the Euclidean
    steps expresses 1 = a(x)*s(x) + m(x)*t(x), so s(x) mod m(x) is the
    inverse. This is the textbook route (mirrors extended_gcd used for
    RSA's modular inverse in encryption_rsa.py) rather than a table
    lookup, to demonstrate the algorithm explicitly.
    """
    if a == 0:
        raise ZeroDivisionError("0 has no multiplicative inverse in GF(2^8)")

    def poly_deg(p: int) -> int:
        return p.bit_length() - 1

    def poly_divmod(num: int, den: int):
        """Polynomial long division over GF(2); returns (quotient, remainder) as ints."""
        if den == 0:
            raise ZeroDivisionError
        quotient = 0
        deg_den = poly_deg(den)
        rem = num
        while rem != 0 and poly_deg(rem) >= deg_den:
            shift = poly_deg(rem) - deg_den
            quotient ^= (1 << shift)
            rem ^= (den << shift)
        return quotient, rem

    # Extended Euclid over GF(2)[x]: track (old_r, old_s) and (r, s) such that
    # old_r = old_s * a  (mod behavior enforced by working mod field modulus)
    old_r, r = AES_MODULUS, a
    old_s, s = 0, 1
    while r != 0:
        q, rem = poly_divmod(old_r, r)
        old_r, r = r, rem
        # old_s, s = s, old_s ^ gf_poly_mult(q, s)  -- multiply-without-reduction needed here
        old_s, s = s, old_s ^ _poly_mult_no_reduce(q, s)
    # old_r should now be 1 (gcd); old_s is s(x) with a*s(x) = 1 (mod m(x))
    inv = old_s & 0xFF
    return inv


def _poly_mult_no_reduce(a: int, b: int) -> int:
    """Plain GF(2)-polynomial multiplication with NO modular reduction (used inside the
    extended Euclidean algorithm, where intermediate values are not field elements)."""
    result = 0
    shift = 0
    bb = b
    while bb:
        if bb & 1:
            result ^= (a << shift)
        bb >>= 1
        shift += 1
    return result


def generate_sbox() -> list[int]:
    """
    Derive the AES S-box mathematically (rather than hardcoding the
    standard 256-entry lookup table).

    The AES S-box is defined, for each byte b, as:
        1. Compute b' = multiplicative inverse of b in GF(2^8)
           (by convention 0 maps to 0, since 0 has no inverse).
        2. Apply the affine transformation over GF(2):
               s_i = b'_i XOR b'_{(i+4) mod 8} XOR b'_{(i+5) mod 8}
                     XOR b'_{(i+6) mod 8} XOR b'_{(i+7) mod 8} XOR c_i
           where c = 0x63 (01100011) is a fixed constant, applied
           bitwise, i indexes bits 0 (LSB) to 7 (MSB).

    This nonlinear inverse step is what gives AES its resistance to
    linear/differential cryptanalysis; the affine step destroys
    simple algebraic structure (fixed points) that inversion alone
    would leave behind.
    """
    sbox = [0] * 256
    c = 0x63
    for b in range(256):
        inv = gf_inverse(b) if b != 0 else 0
        s = 0
        for i in range(8):
            bit = (
                ((inv >> i) & 1)
                ^ ((inv >> ((i + 4) % 8)) & 1)
                ^ ((inv >> ((i + 5) % 8)) & 1)
                ^ ((inv >> ((i + 6) % 8)) & 1)
                ^ ((inv >> ((i + 7) % 8)) & 1)
                ^ ((c >> i) & 1)
            )
            s |= (bit << i)
        sbox[b] = s
    return sbox


def generate_inverse_sbox(sbox: list[int]) -> list[int]:
    """Build the inverse S-box: inv_sbox[sbox[b]] = b, used by AES decryption."""
    inv = [0] * 256
    for b, s in enumerate(sbox):
        inv[s] = b
    return inv


SBOX = generate_sbox()
INV_SBOX = generate_inverse_sbox(SBOX)

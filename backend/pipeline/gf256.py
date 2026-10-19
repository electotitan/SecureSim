"""Finite-field arithmetic over GF(2^8) and derivation of the AES S-box.

Everything here is implemented from first principles (no crypto/ECC libraries).

Mathematical background
-----------------------
GF(2^8) is the set of polynomials of degree <= 7 with coefficients in GF(2) = {0, 1}.
A byte b7 b6 ... b0 is identified with  b7*x^7 + ... + b1*x + b0.

* Addition is coefficient-wise XOR (1 + 1 = 0), so  a + b = a XOR b.
* Multiplication is polynomial multiplication over GF(2) followed by reduction modulo
  the AES irreducible polynomial

        m(x) = x^8 + x^4 + x^3 + x + 1      (hex 0x11B).

  Because m(x) is irreducible over GF(2), the nonzero elements form a multiplicative
  group of order 255, so every nonzero a has a unique inverse with  a * a^-1 = 1.
* Inversion is computed with the extended Euclidean algorithm over GF(2)[x]:
  running Euclid on (m(x), a(x)) yields  s(x)*m(x) + t(x)*a(x) = gcd = 1, hence
  t(x) = a(x)^-1 (mod m(x)).  A separate reference (a^254, by Fermat/Lagrange in a
  group of order 255) is provided for cross-checking.
* The AES S-box is  S(a) = Affine(a^-1)  with  0^-1 := 0 by convention, where
  (FIPS-197, eq. 5.1)

        b'_i = b_i XOR b_(i+4) XOR b_(i+5) XOR b_(i+6) XOR b_(i+7) XOR c_i   (indices mod 8)

  and c = 0x63.  The table is *computed*, never stored.

Byte/bit convention: bit i (value 2^i) of an int is the coefficient of x^i.
"""
from __future__ import annotations

from functools import lru_cache

AES_POLY: int = 0x11B          # x^8 + x^4 + x^3 + x + 1
AFFINE_CONSTANT: int = 0x63    # c in FIPS-197 eq. 5.1
FIELD_ORDER: int = 256


# --------------------------------------------------------------------------- validation
def _check_element(x: int, name: str = "value") -> None:
    """Raise if ``x`` is not an integer in [0, 255]."""
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"{name} must be an int, got {type(x).__name__}")
    if not 0 <= x <= 0xFF:
        raise ValueError(f"{name} must be in the range 0..255, got {x}")


# --------------------------------------------------------------------------- GF(2)[x] helpers
def poly_degree(p: int) -> int:
    """Degree of the GF(2) polynomial encoded by ``p`` (-1 for the zero polynomial)."""
    if p < 0:
        raise ValueError("polynomial must be non-negative")
    return p.bit_length() - 1


def poly_multiply_gf2(a: int, b: int) -> int:
    """Carry-less (polynomial) product over GF(2); no reduction is applied."""
    if a < 0 or b < 0:
        raise ValueError("polynomials must be non-negative")
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        b >>= 1
    return result


def poly_divmod_gf2(a: int, b: int) -> tuple[int, int]:
    """Polynomial long division over GF(2): returns (q, r) with a = q*b + r, deg r < deg b."""
    if b == 0:
        raise ZeroDivisionError("division by the zero polynomial")
    if a < 0 or b < 0:
        raise ValueError("polynomials must be non-negative")
    deg_b = poly_degree(b)
    quotient = 0
    while a and poly_degree(a) >= deg_b:
        shift = poly_degree(a) - deg_b
        quotient ^= 1 << shift
        a ^= b << shift
    return quotient, a


# --------------------------------------------------------------------------- field operations
def gf_add(a: int, b: int) -> int:
    """Addition (and subtraction) in GF(2^8) is XOR."""
    _check_element(a, "a")
    _check_element(b, "b")
    return a ^ b


def gf_multiply(a: int, b: int) -> int:
    """Multiply two field elements: polynomial product over GF(2) reduced mod 0x11B.

    Uses shift-and-add: for each set bit of ``b`` add (XOR) the current multiple of ``a``;
    after each step multiply ``a`` by x and, if the degree reaches 8, subtract m(x).
    """
    _check_element(a, "a")
    _check_element(b, "b")
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1                      # multiply by x
        if a & 0x100:                # degree 8 appeared: reduce modulo m(x)
            a ^= AES_POLY
        b >>= 1
    return result


def gf_pow(a: int, n: int) -> int:
    """Compute a^n in GF(2^8) by square-and-multiply (n >= 0; 0^0 is defined as 1)."""
    _check_element(a, "a")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("exponent must be a non-negative int")
    result = 1
    base = a
    while n:
        if n & 1:
            result = gf_multiply(result, base)
        base = gf_multiply(base, base)
        n >>= 1
    return result


def gf_inverse_with_trace(a: int) -> tuple[int, list[dict]]:
    """Invert ``a`` with the extended Euclidean algorithm over GF(2)[x].

    Maintains the invariant  t_i * a  ==  r_i  (mod m(x))  starting from
    (r0, t0) = (m(x), 0) and (r1, t1) = (a, 1).  Each step divides r_{i-1} by r_i:
        r_{i-1} = q_i * r_i + r_{i+1},      t_{i+1} = t_{i-1} XOR q_i * t_i.
    When the remainder reaches 0, r_prev = gcd(m, a) = 1 and t_prev = a^-1.

    Returns (inverse, steps) where ``steps`` is a JSON-compatible list of integer dicts
    for visualisation.  Raises ValueError for a == 0 (zero has no inverse).
    """
    _check_element(a, "a")
    if a == 0:
        raise ValueError("0 has no multiplicative inverse in GF(2^8)")

    r_prev, r_curr = AES_POLY, a
    t_prev, t_curr = 0, 1
    steps: list[dict] = []
    while r_curr != 0:
        q, r_next = poly_divmod_gf2(r_prev, r_curr)
        t_next = t_prev ^ poly_multiply_gf2(q, t_curr)
        steps.append({
            "dividend": r_prev, "divisor": r_curr, "quotient": q,
            "remainder": r_next, "t_prev": t_prev, "t_curr": t_curr, "t_next": t_next,
        })
        r_prev, r_curr = r_curr, r_next
        t_prev, t_curr = t_curr, t_next

    # Defensive checks: only fail if m(x) were reducible or the code were wrong.
    if r_prev != 1:
        raise ArithmeticError("gcd(m(x), a) != 1; modulus is not irreducible")
    inverse = t_prev
    if not 0 < inverse <= 0xFF:
        raise ArithmeticError("extended Euclid produced an out-of-range inverse")
    return inverse, steps


def gf_inverse(a: int) -> int:
    """Multiplicative inverse of a nonzero field element (extended Euclid over GF(2))."""
    return gf_inverse_with_trace(a)[0]


def gf_inverse_by_exponentiation(a: int) -> int:
    """Independent reference inverse: a^254 = a^-1 because the group has order 255."""
    _check_element(a, "a")
    if a == 0:
        raise ValueError("0 has no multiplicative inverse in GF(2^8)")
    return gf_pow(a, 254)


# --------------------------------------------------------------------------- S-box
def affine_transform(b: int) -> int:
    """AES affine transformation (FIPS-197 eq. 5.1), applied bit by bit."""
    _check_element(b, "b")
    out = 0
    for i in range(8):
        bit = (
            ((b >> i) & 1)
            ^ ((b >> ((i + 4) % 8)) & 1)
            ^ ((b >> ((i + 5) % 8)) & 1)
            ^ ((b >> ((i + 6) % 8)) & 1)
            ^ ((b >> ((i + 7) % 8)) & 1)
            ^ ((AFFINE_CONSTANT >> i) & 1)
        )
        out |= bit << i
    return out


def generate_sbox() -> list[int]:
    """Derive the 256-entry AES S-box:  S(a) = Affine(a^-1), with 0^-1 := 0."""
    return [affine_transform(gf_inverse(a) if a else 0) for a in range(FIELD_ORDER)]


@lru_cache(maxsize=1)
def sbox_table() -> tuple[int, ...]:
    """Cached immutable S-box (computed once by ``generate_sbox``; never hardcoded)."""
    return tuple(generate_sbox())

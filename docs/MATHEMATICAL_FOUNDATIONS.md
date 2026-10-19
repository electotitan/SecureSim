# Mathematical Foundations

> Grows stage by stage. Current sections: GF(2^8) and the AES S-box (Checkpoint 2).

## 1. The finite field GF(2^8)

**Elements.** A byte `b7 ... b0` is the polynomial `b7*x^7 + ... + b1*x + b0` with coefficients in GF(2) = {0, 1}.
There are 256 elements.

**Addition** is coefficient-wise XOR, because 1 + 1 = 0 in GF(2). Every element is its own additive inverse.

**Multiplication** is polynomial multiplication over GF(2), reduced modulo the AES irreducible polynomial

    m(x) = x^8 + x^4 + x^3 + x + 1      (0x11B)

*Why irreducible?* If m(x) factored, two nonzero elements could multiply to zero, and not every nonzero element
would be invertible. Irreducibility makes the structure a field. The test suite checks it directly: a reducible
degree-8 polynomial must have a factor of degree <= 4, and none divides m(x).

*Implementation (shift-and-add).* For each set bit of `b`, XOR the current multiple of `a` into the result; then
multiply `a` by x (left shift) and, if degree 8 appears, XOR with 0x11B. Check: `0x57 * 0x83 = 0xC1`
(FIPS-197 section 4.2).

**Structure.** The 255 nonzero elements form a cyclic group of order 255. `0x03` is a generator (verified: its
powers reach all 255 nonzero elements). By Lagrange, `a^255 = 1`, so `a^-1 = a^254`; this is used as an
independent reference inverse.

## 2. Inversion by the extended Euclidean algorithm over GF(2)[x]

Run Euclid on `(m(x), a(x))`. Each step: `r_{i-1} = q_i * r_i + r_{i+1}`, `t_{i+1} = t_{i-1} XOR q_i * t_i`,
starting from `(r0, t0) = (m, 0)` and `(r1, t1) = (a, 1)`. Invariant: `t_i * a = r_i (mod m)`. When the remainder
reaches 0, the last nonzero remainder is `gcd(m, a) = 1` and its `t` is `a^-1`.

Worked example, computed by the implementation, for `a = 0x53`:

| step | division                      | t_next |
|------|-------------------------------|--------|
| 1    | 0x11B = 0x5  * 0x53 + 0x4     | 0x5    |
| 2    | 0x53  = 0x14 * 0x4  + 0x3     | 0x45   |
| 3    | 0x4   = 0x3  * 0x3  + 0x1     | 0xCA   |
| 4    | 0x3   = 0x3  * 0x1  + 0x0     | (stop) |

Remainder 1 appears at step 3, so `0x53^-1 = 0xCA`, and `0x53 * 0xCA = 0x01`. Zero has no inverse and is rejected.

## 3. The AES S-box

`S(a) = Affine(a^-1)`, with the convention `0^-1 := 0`. The affine map (FIPS-197 eq. 5.1) is

    b'_i = b_i XOR b_(i+4) XOR b_(i+5) XOR b_(i+6) XOR b_(i+7) XOR c_i      (indices mod 8), c = 0x63

Equivalent form used as a cross-check in tests: `b XOR rotl(b,1) XOR rotl(b,2) XOR rotl(b,3) XOR rotl(b,4) XOR 0x63`.

*Why these two steps?* Inversion is highly nonlinear, which resists linear and differential cryptanalysis. The
affine map removes the inversion's simple algebraic structure and the fixed point at 0 (`S(0) = 0x63`).
Verified properties: bijection, no fixed points (`S(a) != a`), no opposite fixed points (`S(a) != NOT a`).

Example: `0x53 -> inverse 0xCA -> affine -> 0xED = S(0x53)`.

## Byte/bit convention (used everywhere)

Bit `i` (value 2^i) of an integer is the coefficient of `x^i`. Hex values are written most-significant first.

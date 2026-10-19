# Mathematical Foundations

> Grows stage by stage. Current sections: GF(2^8) and the AES S-box (Checkpoint 2); Huffman coding and binary representation (Checkpoint 3).

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


---

## 4. Source coding: entropy and Huffman codes (Stage 1)

**Model.** A text is a sequence of symbols (here: Unicode code points). With frequency `f_i` of symbol `i` and
`N = sum f_i`, the empirical probability is `p_i = f_i / N`.

**Shannon entropy** `H = - sum p_i log2 p_i` (bits/symbol) is the lower bound on the average length of any
uniquely decodable binary code (source coding theorem). It is 0 for a single-symbol source and `log2 k` for `k`
equally likely symbols.

**Prefix codes.** No codeword is a prefix of another, so a bitstream decodes uniquely without separators. Codeword
lengths satisfy the Kraft inequality `sum 2^(-l_i) <= 1`; Huffman codes satisfy it with equality (they are
*complete*) whenever there are at least two symbols.

**Average length and efficiency.** `L = sum p_i l_i`, and `eta = H / L * 100 %` (defined only for `L > 0`; the API
returns `null` otherwise). For Huffman codes `H <= L < H + 1`. `eta = 100 %` exactly when every `p_i` is a power
of 1/2 (dyadic).

**Algorithm.** Put one leaf per symbol into a min-heap keyed by weight. Repeatedly remove the two lightest nodes,
join them under a new parent whose weight is their sum, and reinsert it. The last node is the root; the left edge
is bit `0`, the right edge bit `1`. Complexity `O(k log k)` for `k` distinct symbols.

**Why it is optimal.** Let `a`, `b` be the two least probable symbols.
1. *Exchange argument.* Some optimal prefix code has `a` and `b` as sibling leaves at maximum depth: if a more
   probable symbol were deeper than a less probable one, swapping them cannot increase `L`; the deepest leaf must
   have a sibling (otherwise its code could be shortened).
2. *Reduction.* Merging `a, b` into one symbol of weight `p_a + p_b` yields a problem on `k - 1` symbols whose
   optimal cost differs from the original by exactly `p_a + p_b`, independent of the rest of the tree.
3. *Induction* on `k` shows the greedy merge is optimal.

Scope of the claim: optimal among symbol-by-symbol binary prefix codes for the given probabilities. It does not use
context, and the cost of sending the code table is *not* included in the reported encoded length.

**Verification beyond known examples.** The test suite compares Huffman's total cost against an exhaustive search
over all complete prefix codes (for alphabets of 2 to 6 symbols, 150 random weight sets) and reproduces the
classic textbook total of 224 bits for weights 45, 13, 12, 16, 9, 5.

**Conventions.**
* Empty input: empty bitstring, empty table.
* One distinct symbol: code `"0"` (length 1). Then `H = 0`, `L = 1`, `eta = 0 %`: the source carries no
  information, but one bit per symbol keeps the stream length equal to the message length.
* Deterministic ties: leaves are numbered by `(frequency, code point)`, internal nodes continue the numbering, the
  heap key is `(weight, number)`, the first popped node becomes the left child. The code table therefore depends
  only on the symbol counts, not on the text order.
* A symbol is one code point, so a combining sequence such as `e` + U+0301 counts as two symbols.

Worked example `ABBCCC` (computed by the running system): `p = (1/6, 2/6, 3/6)`, `H = 1.4591`, codes
`C=0, A=10, B=11`, `L = 1.5`, `eta = 97.28 %`, bitstring `101111000` (9 bits versus 48 bits of UTF-8).

## 5. Binary representation (Stage 2)

Two distinct operations are easy to confuse:

* **Character to integer** (Unicode): `e-acute` is the abstract code point U+00E9 = 233.
* **Text encoding** (UTF-8): each code point becomes 1 to 4 bytes.

| code point range   | byte pattern                              |
|--------------------|-------------------------------------------|
| U+0000 to U+007F   | `0xxxxxxx`                                |
| U+0080 to U+07FF   | `110xxxxx 10xxxxxx`                       |
| U+0800 to U+FFFF   | `1110xxxx 10xxxxxx 10xxxxxx`              |
| U+10000 to U+10FFFF| `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`     |

So `e-acute` (233) is the two bytes `0xC3 0xA9`, not the single byte `0xE9`. Each byte is written as exactly eight
bits, most significant bit first. Decoding requires complete bytes and strictly valid UTF-8: lone continuation
bytes, overlong forms such as `C0 80`, truncated sequences and encoded surrogates are rejected. Lone surrogates
cannot be encoded at all.

An arbitrary byte stream (for example ciphertext later in the pipeline) is *not* generally valid UTF-8, which is why
the pipeline never applies text decoding to anything except the final, verified plaintext.

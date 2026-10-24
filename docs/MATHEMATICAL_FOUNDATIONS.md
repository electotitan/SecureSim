# Mathematical Foundations

> Grows stage by stage. Current sections: GF(2^8) and the AES S-box (Checkpoint 2); Huffman coding and binary representation (Checkpoint 3); LZ77 compression (Checkpoint 4); AES-128 (Checkpoint 5); RSA key exchange (Checkpoint 6); packetization (Checkpoint 7); CRC-32 (Checkpoint 8).

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


---

## 6. Compression: LZ77 (Stage 3)

**Idea.** Real data repeats itself. LZ77 (Ziv and Lempel, 1977) replaces a repeated substring by a *back-reference*
into a sliding **search window** of data already processed. A **lookahead buffer** holds the bytes still to be coded.

**Token convention** (`(offset, length, next_byte)`):

| field       | meaning |
|-------------|---------|
| `offset`    | distance backward from the current output position; `1 <= offset <= min(window, position)` for a match, `0` for a literal |
| `length`    | number of bytes copied from `output[position - offset :]` (0 for a literal) |
| `next_byte` | the literal byte after the match, as `bytes` of length 1; `b""` only in the **end-of-input token** (a match that reaches the last byte), so no artificial padding is ever needed |

A token covers `length + len(next_byte)` bytes, and the encoder keeps that at most `lookahead_size`. So a match followed
by a literal has `length <= lookahead_size - 1`, while an end-of-input match may use the whole lookahead.

**Overlapping matches.** If `offset < length`, the source region runs into bytes the same token is producing.
Example `aaaa`: literal `(0,0,'a')`, then `(1,3,b"")`: copy 1 byte back, 3 times; each copied byte is the one just written.
Decompression therefore copies **one byte at a time**. When `offset == length` the source ends exactly where the output
begins, which is *not* an overlap.

**Search policy** (deterministic): longest match whose start lies in the window; among equal lengths, the **smallest
offset**. This is a greedy parse: it is simple and fast but not guaranteed to give the smallest possible token list
overall (an optimal parse would need dynamic programming).

Why matches can be binary-searched: if a match of length `L` exists at some offset, one of length `L - 1` exists at the
same offset, so existence is monotone in `L`. The fast engine uses this; a readable brute-force engine implements the same
rule, and the tests require identical token output on 400 random cases plus larger inputs.

**Serialised size (token count is not a size).** Sizes are measured on a concrete fixed-width format:
`offset_bits = W.bit_length()`, `length_bits = L.bit_length()`, a 1-bit "next present" flag, 8 bits if present, packed
into bytes, plus 1 header byte holding the number of padding bits. `W` and `L` are agreed out of band. Defaults
`W = 4096, L = 18` give 13 + 5 + 1 + 8 = **27 bits for a literal that represents one byte**.

    compression ratio  R = original size / compressed size          (R < 1 means expansion)
    space saved           = (1 - compressed / original) * 100 %     (negative means expansion)

Consequences, all verified by tests:
* Short or non-repeating input **expands**: `aaaa` (4 bytes) becomes 7 bytes (R = 0.571, -75 %).
* Two token lists with the same length can have different byte sizes (end-of-input tokens carry no byte).
* Empty input has no tokens, serialises to `b""`, and R is undefined (`null`).

**Complexity.** Reference engine: O(n * W * L) worst case. Fast engine: O(n * log L) substring searches over a window of
at most W bytes, with a small constant because the search runs in C. Both are implementation-dependent, not universal bounds.

**Theory versus this implementation.** Sliding-window LZ77 with an unbounded window is asymptotically optimal for
stationary ergodic sources. This implementation uses a 4 KiB window and fixed-width tokens, so it does *not* achieve
that and is frequently worse than entropy coding on short or noisy data.

### Observed comparison (single seeded run; observations, not general laws)

Measured with the running code (`random.Random(1)`, defaults `W = 4096, L = 18`; Huffman figures are the payload only,
code-table cost not included):

| input (n chars)                         | H (bits/char) | Huffman payload | LZ77 compressed | LZ77 ratio |
|-----------------------------------------|--------------:|----------------:|----------------:|-----------:|
| iid, P(a) = 0.9, P(b) = 0.1 (2000)      | 0.466         | 250.0 B (1.000 b/char) | 419 B    | 4.77 |
| iid uniform over {a, b} (2000)          | 1.000         | 250.0 B         | 649 B           | 3.08 |
| `abc` repeated (2001)                   | 1.585         | 416.9 B         | 385 B           | 5.20 |
| repeated English sentence (2025)        | 4.397         | 1130.6 B        | 480 B           | 4.22 |
| iid uniform lowercase letters (2000)    | 4.689         | 1181.5 B        | 2538 B          | **0.79 (expands)** |

What this illustrates: a *low-entropy* source (first row, H = 0.466) is not automatically compressed well by every method.
Huffman codes whole symbols, so it cannot go below 1 bit/char even though the entropy is 0.466. LZ77 reaches about 1.68
bits/char here, and neither approaches the entropy bound. On structureless data (last row) LZ77 expands it.


---

## 7. AES-128 encryption and decryption (Stages 4B and 11)

> **Educational implementation only.** It is not constant-time, CBC gives no integrity protection, and padding errors
> are distinguishable (a padding-oracle risk). Never use it to protect real data; use a vetted library and an
> authenticated mode (for example AES-GCM) in real systems.

**Structure.** AES-128 is a substitution-permutation network on a 16-byte block with a 16-byte key and 10 rounds:
an initial AddRoundKey, 9 full rounds (SubBytes, ShiftRows, MixColumns, AddRoundKey) and a final round without
MixColumns. Decryption applies the inverse operations with the round keys in reverse order.

**Byte order (used everywhere, including the visualisations).** A block `b[0..15]` is loaded into the 4x4 state column by
column: `state[r][c] = b[r + 4c]`, so bytes 0..3 form column 0. Round keys use the same order, and byte `i` of a round key
is XORed into `state[i mod 4][i div 4]`.

**SubBytes (confusion).** `S(a) = Affine(a^-1)` in GF(2^8) with `0^-1 := 0` (section 3). The S-box is *derived* by the
code, not stored. Inversion supplies the non-linearity; the affine map hides the algebraic structure. Example from the
running code: `0x53 -> inverse 0xca -> S = 0xed`.

**ShiftRows (diffusion across columns).** Row `r` is rotated left by `r` positions. The inverse rotates right.

**MixColumns (diffusion within a column).** Each column is multiplied by the circulant matrix over GF(2^8)

    | 2 3 1 1 |
    | 1 2 3 1 |        (equivalently a(x) = 3x^3 + x^2 + x + 2  modulo x^4 + 1)
    | 1 1 2 3 |
    | 3 1 1 2 |

The inverse matrix is `[14 11 13 9; 9 14 11 13; 13 9 14 11; 11 13 9 14]`; the tests verify that the two multiply to the
identity over GF(2^8) and that `InvMixColumns(MixColumns(x)) = x` on 300 random states. All multiplication tables
are computed from `gf_multiply`.

**AddRoundKey.** XOR with the round key (addition in GF(2^8)); it is its own inverse.

**Key expansion.** For `i = 4 .. 43`: `w[i] = w[i-4] XOR T(w[i-1])` where for `i` divisible by 4
`T = SubWord(RotWord(.)) XOR Rcon[i/4]`, otherwise `T` is the identity. `Rcon[j] = x^(j-1)` in GF(2^8)
(`01 02 04 08 10 20 40 80 1b 36`, computed by repeated multiplication by 2).

### Worked example (FIPS-197 Appendix C.1, produced by the running code)

Key `000102030405060708090a0b0c0d0e0f`, plaintext `00112233445566778899aabbccddeeff`, ciphertext
`69c4e0d86a7b0430d8cdb78070b4c55a`. State bytes below are in the column-major order described above.

| step (round 1)                     | state |
|------------------------------------|-------|
| after initial AddRoundKey          | `00102030405060708090a0b0c0d0e0f0` |
| SubBytes                           | `63cab7040953d051cd60e0e7ba70e18c` |
| ShiftRows                          | `6353e08c0960e104cd70b751bacad0e7` |
| MixColumns                         | `5f72641557f5bc92f7be3b291db9f91a` |
| round key 1                        | `d6aa74fdd2af72fadaa678f1d6ab76fe` |
| AddRoundKey (start of round 2)     | `89d810e8855ace682d1843d8cb128fe4` |

One MixColumns output byte, worked in GF(2^8) (column 0, row 0): `{02}*{63} xor {03}*{53} xor {01}*{e0} xor {01}*{8c} = c6 xor f5 xor e0 xor 8c = 0x5f`.

### CBC mode and PKCS#7 padding

CBC: `C_0 = IV`, `C_i = E_K(P_i XOR C_(i-1))`; decryption `P_i = D_K(C_i) XOR C_(i-1)`. The IV is not secret but should be
unpredictable and not reused with the same key. The IV is not prepended to the output by `aes_cbc_encrypt`; the API
returns it separately.

PKCS#7 **always** appends `n` bytes of value `n` (`1 <= n <= 16`), so a block-aligned message receives a whole extra block
(otherwise unpadding would be ambiguous). An empty message therefore encrypts to exactly one block. Unpadding rejects: a
zero or >16 length byte, inconsistent padding bytes, and data that is empty or not a multiple of 16.

Properties demonstrated by the tests:
* identical plaintext blocks give different ciphertext blocks (unlike ECB);
* **malleability**: flipping bit `b` of ciphertext block `i` garbles plaintext block `i` and flips exactly bit `b` of block
  `i+1`, with all other blocks unaffected. CBC detects nothing by itself;
* a wrong IV damages only the first plaintext block; a wrong key gives garbage or (usually) a padding error, which
  cannot be relied on as an integrity check.

### Verification evidence

* FIPS-197 Appendix C.1 (the gating vector) and Appendix B block vectors, encrypt and decrypt.
* FIPS-197 round-1 intermediate states, key-expansion words (Appendix A.1) and the round-1 key.
* NIST SP 800-38A ECB (F.1.1, first block) and CBC (F.2.1, four blocks) vectors.
* Known MixColumns columns (`db135345 -> 8e4da1bc`, `f20a225c -> 9fdc589d`, ...).
* Round trips on 400 random key/block pairs, CBC for many message lengths including 0 and exact multiples of 16.
* Experimental sanity check (not a security proof): over 200 seeded samples a single flipped plaintext bit changed
  between 56 and 72 of 128 ciphertext bits on average (the full avalanche experiment is a later checkpoint).

Only AES-128 is implemented. Requests for AES-192/256 receive HTTP 501 (`unsupported_operation`).


---

## 8. RSA key exchange (Stage 5)

> **Educational and insecure.** This is *textbook* RSA: no padding (no OAEP), tiny keys. Every API response is labelled
> `insecure: true`. Real RSA needs a modulus of at least 2048 bits, OAEP padding and a vetted library. Even the 160-bit moduli
> used below to carry a 128-bit AES key can be factored almost instantly.

**Key generation.** Choose distinct odd primes `p`, `q`; `n = p*q`; `phi(n) = (p-1)(q-1)`. Choose `e` with `1 < e < phi` and
`gcd(e, phi) = 1` (default 65537, falling back to 257, 17, 5, 3 when `phi` is too small or shares a factor). Compute
`d = e^-1 (mod phi)` with the extended Euclidean algorithm, so `e*d = 1 + k*phi`. Here `bits` means the **modulus** size
(even, 12 to 1024); `p` and `q` have `bits/2` bits with their top two bits set, so `n` has exactly `bits` bits.

**Encryption and decryption.** For an integer `0 <= m < n`: `c = m^e mod n` and `m = c^d mod n`.

**Why decryption works (for every m).** For a prime `r` in `{p, q}`: if `r` does not divide `m`, Fermat's little theorem gives
`m^(r-1) = 1 (mod r)`; since `(r-1)` divides `e*d - 1 = k*phi`, `m^(ed) = m * (m^(r-1))^j = m (mod r)`. If `r` divides `m`, both sides
are `0 (mod r)`. Because `n` is squarefree, the Chinese remainder theorem gives `m^(ed) = m (mod n)`, including for `m` not
coprime to `n` (the tests check all 143 residues of `n = 11*13` and all 3233 of `n = 61*53`).

### Worked example (the classic one; produced by the running code)

`p = 61, q = 53, e = 17`: `n = 3233`, `phi = 3120`, `d = 2753`.

Computing `d` by the extended Euclidean algorithm on `(17, 3120)` (columns: quotient, remainder, and the coefficients
`x, y` after each step with `17*x + 3120*y = remainder`):

| step | quotient | remainder | x | y |
|-----:|---------:|----------:|--:|--:|
| 1 | 0 | 17 | 1 | 0 |
| 2 | 183 | 9 | -183 | 1 |
| 3 | 1 | 8 | 184 | -1 |
| 4 | 1 | 1 | -367 | 2 |
| 5 | 8 | 0 | 3120 | -17 |

The last nonzero remainder is `1`, and `x mod 3120 = 2753`, so `d = 2753` and `17 * 2753 mod 3120 = 1`.

**Square-and-multiply** computes `65^17 mod 3233`. `17` is `10001` in binary. The leading `1` loads the base (`65`); each later
bit squares, and also multiplies by the base when the bit is `1`:

| step | bit | before | after square | multiply? | after multiply |
|-----:|----:|-------:|-------------:|:---------:|---------------:|
| 1 | 0 | 65 | 992 | no | 992 |
| 2 | 0 | 992 | 1232 | no | 1232 |
| 3 | 0 | 1232 | 1547 | no | 1547 |
| 4 | 1 | 1547 | 789 | yes | 2790 |

Ciphertext `c = 2790` after 4 squarings and 1 multiplication (versus 16 multiplications for naive
repeated multiplication). Decryption computes `2790^2753 mod 3233` in 11 squarings and 4 multiplications and returns `65`.
In general `exponent_bits - 1` squarings plus `popcount - 1` multiplications are needed: `O(log e)` modular multiplications.

### Miller-Rabin primality testing

Write `n - 1 = 2^s * d` with `d` odd. A base `a` is a **witness** that `n` is composite if `a^d != 1` and `a^(d*2^r) != -1 (mod n)`
for all `0 <= r < s`. For a prime no witness exists; for an odd composite at least 3/4 of all bases are witnesses.

Strategy implemented (and why):
1. Trial division by the primes below 100 (found by a sieve). If no factor is found and `n < 101^2 = 10201`, `n` is prime.
2. For `n < 3,317,044,064,679,887,385,961,981` the first 13 prime bases (2, 3, ..., 41) are a **proven** deterministic test
   (Sorenson and Webster, 2015).
3. For larger `n` the same 13 bases are used plus 20 random bases. Each random base lets a composite through with probability
   at most 1/4, so a composite is accepted with probability below `4^-20 = 9.095e-13`; a prime is always accepted.

Two concrete illustrations (computed by the code):
* `n = 3215031751 = 151 * 751 * 28351` has `n - 1 = 2^1 * 1607515875`. It is a strong pseudoprime to the bases 2, 3, 5, 7
  (`a^d mod n` is 2: 1, 3: 3215031750, 5: 1, 7: 3215031750, i.e. `1` or `n - 1`), but base 11 gives `2129160099`, which exposes it. A test with only the first four primes would be fooled.
  Its factors are all above 100, so trial division cannot reject it.
* `n = 3317044064679887385961981` (the deterministic bound itself) is the smallest composite that passes **all 13** fixed
  bases. The random bases catch it (witness 305178576366315321245234 in a seeded run); with zero random rounds it would be wrongly accepted
  (checked by a test), which is why those rounds exist.

Carmichael numbers (`a^(n-1) = 1` for all `a` coprime to `n`) fool the Fermat test but not Miller-Rabin; the tests use
Carmichael numbers whose prime factors are all above 100 (for example `211*421*631 = 56052361`) so that Miller-Rabin itself, not trial
division, must reject them.

### Key transport and why a 128-bit key needs a bigger modulus

Raw RSA only handles integers smaller than `n`. A 16-byte AES key is an integer up to `2^128 - 1`, so the modulus must exceed
`2^128` (at least 129 bits; the API requires at least 130 because moduli have an even number of bits and defaults to 160).
`rsa_wrap_key` **refuses** a key whose integer value is `>= n` (tested at exactly `n`, `n-1`, `n+1`) instead of silently
reducing it modulo `n`, which would make the key unrecoverable. Decryption also refuses an integer that does not fit the key
length. The alternative shown by `mode: "integer"` is a deliberately simplified demonstration with a small integer, **not** real
key transport.

### Textbook-RSA weaknesses (each demonstrated by a test)

* Deterministic: equal messages give equal ciphertexts (no probabilistic padding).
* Malleable: `E(a) * E(b) mod n = E(a*b mod n)`.
* `0`, `1` and `n - 1` are fixed points; encryption is a permutation of the residues mod `n`.
* Tiny keys: factoring `n` reveals `phi` and therefore `d`.

OAEP is **not** implemented: it would require the mask-generation function and a hash implemented from first principles.

### Implementation restrictions (enforced by tests)

Modular exponentiation never uses Python's built-in three-argument `pow` (an AST-based test checks the RSA modules and `main.py`);
`pow` appears only in tests, as an independent reference, together with `math.gcd`. No `math`, `hashlib`, `Crypto`,
`cryptography` or `sympy` import is allowed in the RSA modules, which are also independent of the AES message path.

Measured with the running code: a 160-bit key pair is generated in about a millisecond and a 512-bit key pair in about 0.04 s
(including the HTTP round trip and a 54 KB response).


---

## 9. Packetization (Stage 6)

**Purpose.** A byte string (in the full pipeline: the AES ciphertext) is cut into fixed-size chunks. Each chunk travels in a
packet that also carries what the receiver needs to rebuild the message even when packets arrive late, out of order, twice, or
not at all.

### Wire format (big-endian; 14-byte header, then the payload)

| offset | size | field | meaning |
|-------:|-----:|-------|---------|
| 0 | 1 | `version` | always 1 |
| 1 | 1 | `flags` | reserved, must be 0 |
| 2 | 4 | `sequence` | 0-based packet index (0 .. total-1) |
| 6 | 4 | `total` | number of packets in the message (>= 1) |
| 10 | 2 | `payload_length` | payload bytes that follow (0 .. 65535) |
| 12 | 2 | `header_checksum` | 16-bit Internet checksum of the 14 header bytes |
| 14 | n | payload | the chunk |

`serialized_size = 14 + payload_length` is **not** the payload size; the difference is protocol overhead.
`payload_size` (an argument) is the maximum payload per packet.

### Chunking arithmetic and overhead

For `n` data bytes and payload size `s`: `packet_count = ceil(n / s)` packets, all full except possibly the last (never an empty trailing
packet). An **empty message is one packet with an empty payload and `total = 1`**, not zero packets, so a receiver can tell
"the message is empty" from "nothing arrived".

Overhead per packet is the fixed 14 bytes, so efficiency is `s / (s + 14)` for full packets. Measured with the running code for a
1000-byte message:

| payload size | packets | serialized bytes | overhead | efficiency |
|-------------:|--------:|-----------------:|---------:|-----------:|
| 8 | 125 | 2750 | 63.64 % | 36.36 % |
| 16 | 63 | 1882 | 46.87 % | 53.13 % |
| 32 | 32 | 1448 | 30.94 % | 69.06 % |
| 64 | 16 | 1224 | 18.30 % | 81.70 % |
| 128 | 8 | 1112 | 10.07 % | 89.93 % |
| 256 | 4 | 1056 | 5.30 % | 94.70 % |
| 512 | 2 | 1028 | 2.72 % | 97.28 % |
| 1000 | 1 | 1014 | 1.38 % | 98.62 % |

Small payloads waste most of each packet on headers; very large payloads reduce overhead but lose more data per lost packet. That
trade-off is the reason for the `payload_size` slider in the experiments.

### The header integrity field: the Internet checksum (RFC 1071)

Treat the data as 16-bit big-endian words (append a zero byte if the length is odd), add them with **end-around carry** (any carry out of
bit 16 is added back in; this is one's-complement addition), and take the one's complement of the result. The checksum field is 0 while it
is computed. A receiver sums **all** header words including the stored checksum: the result is `0xFFFF`, so the checksum of the whole header
is 0.

Worked example, packet 0 of the 37-byte message with payload size 10 (header words `0100 0000 0000 0000 0004 000a`, checksum field
`0000`): the sum is `0x0100 + 0x0004 + 0x000a = 0x010e`, and the checksum is `~0x010e = 0xfef1`. The full table (produced by the code):

| sequence | total | payload length | header (hex) | checksum |
|---------:|------:|---------------:|--------------|----------|
| 0 | 4 | 10 | `01000000000000000004000afef1` | `0xfef1` |
| 1 | 4 | 10 | `01000000000100000004000afef0` | `0xfef0` |
| 2 | 4 | 10 | `01000000000200000004000afeef` | `0xfeef` |
| 3 | 4 | 7 | `010000000003000000040007fef1` | `0xfef1` |

Verification: RFC 1071's example (`0001 f203 f4f5 f6f7` gives `0x220d`) and the well-known IPv4 header example (checksum `0xb861`) both pass, and
the implementation is cross-checked against an independent big-integer implementation on 2000 random inputs.

**What it detects.** Every single-bit error: all 112 header bits of four different packets are flipped one at a time and every flip is detected
(either by the parser or by the checksum).

**What it cannot detect (each demonstrated by a test, using the code's own packets).**
* **Reordered words.** Addition is commutative. Swapping the low words of `sequence` and `total` turns packet `3/5` into `5/3` and the header still
  verifies (`header_ok() = True`). Reassembly's *consistency* checks then catch it, not the checksum.
* **Cancelling errors.** Adding 1 to one word and subtracting 1 from another leaves the sum unchanged: packet `3/5` becomes `4/4` and still verifies
  (`header_ok() = True`).
* **Payload damage.** The checksum covers the header only. Replacing `ORIGINAL` with `TAMPERED` leaves a valid header (`header_ok() = True`) and
  reassembly returns the tampered bytes. Payload integrity is the job of the **CRC-32 stage that follows**; a CRC, unlike this sum, detects reordering
  and burst errors. The checksum therefore protects the *framing*, and the CRC will protect the *content*.

### Reassembly rules

Packets may arrive in any order; reassembly sorts by sequence number. It checks, in this **priority order** (a corrupt header makes every other
field untrustworthy):

| priority | condition | error kind |
|---------:|-----------|------------|
| 1 | no packets at all | `no_packets` |
| 2 | any header checksum failure | `header_checksum` |
| 3 | packets disagree about `total` | `inconsistent_total` |
| 4 | a sequence number outside `0 .. total-1` | `invalid_sequence` |
| 5 | the same sequence twice (identical copies can be tolerated; conflicting copies never) | `duplicate` |
| 6 | some sequence numbers never arrived | `missing` |

The non-raising diagnosis (`inspect_packets`) discards packets that fail their checksum before computing totals, duplicates and missing numbers: a
corrupted packet cannot be trusted to fill a slot, and its damaged fields must not create a bogus duplicate (for example a flipped sequence number
colliding with a genuine packet) or hide a missing one.

**Resource safety.** A forged header may claim `total = 4,294,967,295`. Missing and invalid-sequence lists in results are capped at 1,000 entries while
the exact count is always reported, so reassembly never builds a list that large (a test uses exactly this forged header).

### Experiments

The API can perturb the packet set before reassembly, in a fixed, reproducible order: **corrupt headers** (flip one sequence-number bit, keep the old
checksum) → **duplicate** → **drop** (explicit sequences, then random loss) → **shuffle**. Random loss removes each packet independently with
probability `p` using `random.Random(seed)` (one draw per packet, in list order); shuffling uses `random.Random(seed).shuffle`. The expected number of
lost packets is `p * packet_count`, but an individual run, like the channel BER later, does not have to match it exactly.

Measured: a 40,000-byte message split into 40,000 one-byte packets with 20 % loss was packetized, perturbed, diagnosed and reassembled in under 0.7 s
over HTTP (31,850 packets received, 8,150 reported missing).


---

## 10. Error detection: CRC-32 (Stage 7)

**Idea.** A CRC treats a message as a polynomial over GF(2) (the two-element field: addition is XOR, there are no carries) and keeps only the
remainder of a polynomial division. Sender and receiver both compute it; a mismatch means the data changed in transit. Unlike the Internet checksum in
section 9, a CRC is built on *division*, which makes it sensitive to the **order** of the bits and very good at **burst** errors.

**Generator polynomial.** CRC-32 (IEEE 802.3; used by Ethernet, zlib, PNG, gzip):

    G(x) = x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
         = 0x1_04C11DB7        (the "normal" form 0x04C11DB7 drops the implicit x^32 term)

The *reflected* form `0xEDB88320` is its bit reversal (derived in the code, not typed in) and is used because the algorithm shifts the register
right and consumes each byte least significant bit first, which is the order the bits are transmitted.

**Definition.** `CRC = (M(x) * x^32) mod G(x)`, with the standard conventions *init* `0xFFFFFFFF` (register starts as all ones, so leading zero bytes
matter), *reflected input and output* and *xorout* `0xFFFFFFFF` (the result is complemented). The standard check: the ASCII string `123456789` has
CRC-32 `0xCBF43926` (a required test; it passes in the unit tests and over HTTP).

### From polynomial division to a lookup table

One byte is eight bit steps: shift the register right by one, and if the bit that fell out was 1, XOR with `0xEDB88320`. Because the steps are linear,
what a byte does to the register can be precomputed for all 256 byte values: `table[i]` is the result of loading `i` and doing the eight steps, and then
one byte costs `reg = table[(reg ^ byte) & 0xFF] ^ (reg >> 8)`. The table is **derived at import** (`generate_table`), never stored. For `i = 1`
(the code's own steps):

| step | bit shifted out | register |
|-----:|:---------------:|----------|
| 0 | - | `0x00000001` |
| 1 | 1 | `0xedb88320` |
| 2 | 0 | `0x76dc4190` |
| 3 | 0 | `0x3b6e20c8` |
| 4 | 0 | `0x1db71064` |
| 5 | 0 | `0x0edb8832` |
| 6 | 0 | `0x076dc419` |
| 7 | 1 | `0xee0e612c` |
| 8 | 0 | `0x77073096` |

giving `table[1] = 0x77073096`; likewise `table[2] = 0xee0e612c`, `table[255] = 0x2d02ef8d`. The same generator also produces the CRC-32C
(Castagnoli) table from `0x82F63B78`, whose check value `0xE3069283` the tests confirm.

**Three independent computations agree** (tests on every length 0 to 70, on thousands of random inputs, and on all 256 single bytes): the table-driven
version, the bit-by-bit version, and a literal long division of the augmented message in the non-reflected domain, plus Python's `zlib.crc32` used only as
an outside reference in the tests (the implementation imports no `zlib`, `binascii`, `crcmod` or `hashlib`; checked by an AST test).

### Worked example: `123456789`

Initial register `0xFFFFFFFF`; each row applies the table rule to one byte:

| byte # | input | register before | table index | table entry | register after |
|-------:|-------|-----------------|-------------|-------------|----------------|
| 0 | `1` (0x31) | `0xffffffff` | `0xce` | `0x7cdcefb7` | `0x7c231048` |
| 1 | `2` (0x32) | `0x7c231048` | `0x7a` | `0xb0d09822` | `0xb0acbb32` |
| 2 | `3` (0x33) | `0xb0acbb32` | `0x01` | `0x77073096` | `0x77b79c2d` |
| ... | | | | | |

After all nine bytes the register is `0x340bc6d9`; complementing it gives `0xcbf43926`.

### What a CRC-32 detects, and what it cannot

Each claim is either guaranteed by the mathematics or measured by the tests, and the API states which (`guaranteed_detection`).

| error | detected? | evidence |
|-------|-----------|----------|
| any single-bit error | always | every bit of messages up to 64 bytes flipped one at a time |
| any burst of 32 bits or fewer | always | an error polynomial of degree < 32 is never a nonzero multiple of G(x); all lengths 1..32 at every position of a 16-byte message, plus 3,000 random bursts |
| any two-bit error (tested up to 256-bit messages) | always | all 32,640 pairs in a 32-byte message |
| any three-bit error (tested up to 64-bit messages) | always | all 41,664 triples in an 8-byte message |
| a burst longer than 32 bits | almost always, **not guaranteed** | about 1 in 2^32 (2.3e-10) of random patterns escape |
| an error pattern that is a multiple of G(x) | **never** | constructed below |

No claim is made about odd numbers of errors: G(x) has 15 terms, an odd number, so it is not divisible by (x + 1) and parity-based
guarantees do not apply.

**A concrete undetected error.** Bit `p` (in transmission order) of an `n`-bit message is the coefficient of `x^(n-1-p)`. The error polynomial
`G(x) * x^k` is therefore the set of bits `p = n - 1 - (e + k)` for each exponent `e` of G. For the 8-byte message `00 01 02 03 04 05 06 07`
(`n = 64`, `k = 5`) that flips 15 bits spanning 33 positions: `[26, 32, 35, 36, 42, 46, 47, 48, 50, 51, 53, 54, 56, 57, 58]`. The corrupted message is different, yet its CRC is
`0x88aa689f`, exactly the original's. The API reports `guaranteed_detection: false`, `detected: false` for this pattern.

**Detection is not correction.** A mismatch says only that something changed. (Hamming and Reed-Solomon, in the next checkpoints, add the redundancy needed
to locate and repair errors.)

### CRC versus the Internet checksum, and layering

Swapping the 16-bit words of `123456789abcdef0` to `9abc1234def05678` leaves the Internet checksum unchanged (`0x1da6` both times) but changes the CRC
(`0xa85a34a3` against `0xe77851f4`); replacing `ORIGINAL` by `TAMPERED` in a packet payload leaves the packet *header*
checksum valid but the CRC differs. That is why the packet header carries a cheap checksum for *framing* and the CRC, computed over the whole packet, protects
the *content*.

### A CRC is not authentication

CRC is **affine** over GF(2): for equal-length messages `crc(a xor b xor c) = crc(a) xor crc(b) xor crc(c)`. That makes it easy to forge. After four more
bytes the register depends only on the four table indices used, and the top byte of each table entry is distinct, so the four bytes that force **any**
chosen CRC can be computed (`crc32_forge_suffix`). Example (verified with `zlib` over HTTP): the message `Transfer 100 EUR to 1234` was altered to
`Transfer 130 EUR to 1234` and the 4 bytes `7c25016b` appended; the altered message then has the original's CRC `0xa9c88955`.
Anyone can do this, so a CRC protects against accidents, not against an adversary: use a MAC such as HMAC for that.

Also by linearity, appending a CRC (little-endian) to a message gives the constant residue `0x2144df1c` for every message, a standard receiver check that
the tests verify.

Measured with the running code: a 40,000-byte message with a 4,096-bit burst, the full 256-entry table and traces was processed over HTTP in 0.13 s.

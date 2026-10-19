# Mathematical Foundations

> Grows stage by stage. Current sections: GF(2^8) and the AES S-box (Checkpoint 2); Huffman coding and binary representation (Checkpoint 3); LZ77 compression (Checkpoint 4); AES-128 (Checkpoint 5).

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

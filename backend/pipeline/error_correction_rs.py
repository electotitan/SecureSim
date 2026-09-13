"""
error_correction_rs.py — Reed-Solomon error-correcting code over
GF(2^8) (Coding Theory / Finite Field Arithmetic).

SCOPE LIMITATION (documented per project spec): full Reed-Solomon
decoders use Berlekamp-Massey to find the error-locator polynomial for
an arbitrary number of correctable errors. For this educational demo
we instead implement the more directly-derivable
Peterson-Gorenstein-Zierler (PGZ) algorithm, which solves a small
linear system for the error locator polynomial. PGZ is
computationally simple for a *small, fixed* number of correctable
errors but scales poorly (the linear system grows as O(t^2) unknowns
for t errors, solved with an O(t^3) matrix inverse) -- so this
implementation is hardcoded to correct **up to 2 symbol errors**
(t_max = 2), matching the "manageable scope" called out in the spec.
For messages needing to correct more than 2 symbol errors per block,
a Berlekamp-Massey based decoder would be required instead.

Mathematical background
------------------------
An RS(n, k) code over GF(2^8) takes k message symbols and appends
n-k parity symbols, working entirely with codewords viewed as
polynomials over GF(2^8). We build a generator polynomial
    G(x) = (x - alpha^0)(x - alpha^1)...(x - alpha^(n-k-1))
whose roots are consecutive powers of the field's primitive element
alpha (see gf256.py's EXP_TABLE/LOG_TABLE, generator alpha=0x03).
Systematic encoding appends parity = remainder of
(message(x) * x^(n-k)) divided by G(x), so that the *full* codeword
polynomial is always exactly divisible by G(x) -- i.e. every codeword
evaluates to 0 at every root alpha^i of G(x).

Decoding exploits that fact: for a received (possibly corrupted)
codeword R(x) = C(x) + E(x) (E(x) is the unknown "error polynomial",
zero everywhere except at error locations), evaluating R at each root
alpha^i gives
    S_i = R(alpha^i) = C(alpha^i) + E(alpha^i) = 0 + E(alpha^i) = E(alpha^i)
These S_i are the *syndromes* -- if all are zero, no error occurred.
If not, PGZ sets up a linear system (a Vandermonde-like matrix built
from the syndromes) whose solution gives the coefficients of the
*error locator polynomial* Lambda(x), whose roots' reciprocals reveal
exactly which symbol positions are corrupted (Chien search). Once
locations are known, a second small linear system (or, as implemented
here, direct substitution back into the syndrome equations) yields
the error *magnitudes*, and errors are corrected by subtracting
(=XOR-ing, in GF(2^8)) them out.
"""

from __future__ import annotations
from . import gf256

MAX_CORRECTABLE_ERRORS = 2  # PGZ scope limit for this implementation


class RSPolynomial:
    """Thin wrapper: a polynomial over GF(2^8) as a list of coefficients,
    HIGHEST degree first (coeffs[0] is the leading coefficient)."""

    @staticmethod
    def multiply(a: list[int], b: list[int]) -> list[int]:
        result = [0] * (len(a) + len(b) - 1)
        for i, ca in enumerate(a):
            if ca == 0:
                continue
            for j, cb in enumerate(b):
                result[i + j] ^= gf256.gf_multiply_fast(ca, cb)
        return result

    @staticmethod
    def divmod(dividend: list[int], divisor: list[int]) -> tuple[list[int], list[int]]:
        """Polynomial long division over GF(2^8): returns (quotient, remainder)."""
        dividend = dividend[:]
        deg_diff = len(dividend) - len(divisor)
        if deg_diff < 0:
            return [0], dividend
        quotient = [0] * (deg_diff + 1)
        lead_divisor = divisor[0]
        for i in range(deg_diff + 1):
            coef = dividend[i]
            if coef == 0:
                continue
            factor = gf256.gf_divide(coef, lead_divisor)
            quotient[i] = factor
            for j in range(len(divisor)):
                dividend[i + j] ^= gf256.gf_multiply_fast(factor, divisor[j])
        remainder = dividend[deg_diff + 1:]
        # strip leading zeros but keep at least one coefficient
        while len(remainder) > 1 and remainder[0] == 0:
            remainder.pop(0)
        return quotient, remainder


def rs_generator_polynomial(n_parity_symbols: int) -> list[int]:
    """
    Build G(x) = product_{i=0}^{n_parity-1} (x - alpha^i), with roots at
    consecutive powers of the field's primitive element alpha (highest
    degree coefficient first). Since char=2, "x - alpha^i" == "x +
    alpha^i", so coefficients combine via XOR as usual.
    """
    g = [1]  # start with the polynomial "1" (degree 0)
    for i in range(n_parity_symbols):
        root = gf256.gf_pow(gf256.GENERATOR, i)
        # multiply g(x) by (x - root) = (x + root) in GF(2^8): coefficients [1, root]
        g = RSPolynomial.multiply(g, [1, root])
    return g


def rs_encode(message_symbols: list[int], n_parity: int) -> list[int]:
    """
    Systematic Reed-Solomon encoding: append n_parity parity symbols so
    that [message_symbols + parity] forms a codeword exactly divisible
    by the generator polynomial.

    parity = remainder of ( message(x) * x^n_parity ) / G(x)

    Multiplying by x^n_parity shifts the message into the high-order
    positions, leaving room at the low-order end for the parity
    remainder -- this is exactly how systematic CRC/RS codes keep the
    original message bytes visibly unchanged in the codeword.
    """
    if any(not (0 <= s <= 255) for s in message_symbols):
        raise ValueError("all message symbols must be valid bytes (0-255)")
    generator = rs_generator_polynomial(n_parity)
    shifted = message_symbols + [0] * n_parity  # message(x) * x^n_parity
    _, remainder = RSPolynomial.divmod(shifted, generator)
    # pad remainder on the left to exactly n_parity symbols
    parity = [0] * (n_parity - len(remainder)) + remainder
    return message_symbols + parity


def _compute_syndromes(received: list[int], n_parity: int) -> list[int]:
    """
    S_i = R(alpha^i) for i = 0..n_parity-1, evaluated via Horner's
    method over GF(2^8) (R here is treated with received[0] as the
    HIGHEST-degree coefficient, matching rs_encode's convention).
    """
    syndromes = []
    for i in range(n_parity):
        root = gf256.gf_pow(gf256.GENERATOR, i)
        # Horner's method: evaluate polynomial at `root`
        result = 0
        for coeff in received:
            result = gf256.gf_multiply_fast(result, root) ^ coeff
        syndromes.append(result)
    return syndromes


def _solve_linear_system_gf256(matrix: list[list[int]], rhs: list[int]) -> list[int] | None:
    """
    Solve A*x = rhs over GF(2^8) via Gaussian elimination with partial
    pivoting (using GF(2^8) arithmetic throughout: "division" for
    pivoting, XOR for row elimination). Returns None if the matrix is
    singular (no unique solution -- used by PGZ to detect when a
    trial error count doesn't match the actual number of errors).
    """
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            return None  # singular matrix
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot_val = aug[col][col]
        inv_pivot = gf256.gf_inverse(pivot_val)
        aug[col] = [gf256.gf_multiply_fast(v, inv_pivot) for v in aug[col]]

        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [aug[r][j] ^ gf256.gf_multiply_fast(factor, aug[col][j]) for j in range(n + 1)]

    return [aug[i][n] for i in range(n)]


def _peterson_gorenstein_zierler(syndromes: list[int], n_total: int) -> tuple[list[int], list[int]] | None:
    """
    Try to find up to MAX_CORRECTABLE_ERRORS error locations and
    magnitudes using the PGZ algorithm: for a trial error count
    v = MAX_CORRECTABLE_ERRORS down to 1, build the v x v syndrome
    matrix
        M[i][j] = S_{i+j},   i,j = 0..v-1
    and solve M * Lambda = -[S_v, S_{v+1}, ..., S_{2v-1}]^T (all in
    GF(2^8), so "-" is the same as "+"/XOR) for the error-locator
    polynomial's coefficients Lambda_1..Lambda_v. If M is singular,
    fewer errors than v actually occurred; retry with a smaller v.

    Once Lambda is known, find its roots by brute-force evaluation
    over all n_total field-power positions (Chien search), then invert
    each root to get error LOCATIONS, and solve a second linear system
    (Forney-style, done here directly via substitution into the
    syndrome equations) for error MAGNITUDES.

    Returns (error_positions, error_magnitudes) or None if no
    consistent solution is found within MAX_CORRECTABLE_ERRORS.
    """
    n_parity = len(syndromes)

    for v in range(min(MAX_CORRECTABLE_ERRORS, n_parity // 2), 0, -1):
        # Derivation: sigma(x)=1+sigma_1 x+...+sigma_v x^v has sigma(X_k^-1)=0 for
        # each error locator X_k, which (after multiplying by Y_k X_k^{j+v} and
        # summing over k) yields the key equation
        #     sigma_1*S_{j+v-1} + sigma_2*S_{j+v-2} + ... + sigma_v*S_j = S_{j+v}
        # for j = 0..v-1. Row j's coefficients of [sigma_1..sigma_v] are
        # therefore [S_{j+v-1}, S_{j+v-2}, ..., S_j] (descending), NOT simply
        # S_{i+j} -- getting this column order backwards silently produces a
        # solvable-looking but WRONG error locator for v >= 2.
        matrix = [[syndromes[j + v - 1 - col] for col in range(v)] for j in range(v)]
        rhs = [syndromes[j + v] for j in range(v)]
        solution = _solve_linear_system_gf256(matrix, rhs)
        if solution is None:
            continue  # singular -> try fewer assumed errors

        # Lambda(x) = 1 + Lambda_1*x + ... + Lambda_v*x^v (solution = [Lambda_1..Lambda_v])
        lambda_coeffs = [1] + solution  # index 0 = constant term = 1

        # Chien search: find roots of Lambda by trying every nonzero field element
        # X_k = alpha^{-k} for candidate error position k (0-indexed from the low end).
        error_positions = []
        for pos in range(n_total):
            x_inv = gf256.gf_pow(gf256.GENERATOR, pos)  # candidate alpha^pos
            # evaluate Lambda at x = alpha^{-pos}, i.e. at the inverse of x_inv
            x = gf256.gf_inverse(x_inv) if x_inv != 0 else 0
            val = 0
            power = 1
            for coeff in lambda_coeffs:
                val ^= gf256.gf_multiply_fast(coeff, power)
                power = gf256.gf_multiply_fast(power, x)
            if val == 0:
                error_positions.append(pos)

        if len(error_positions) != v:
            continue  # Chien search didn't find exactly v roots -> inconsistent, try smaller v

        # Solve for error magnitudes Y_k via the defining syndrome equations:
        #   S_i = sum_k Y_k * (alpha^{pos_k})^i    for i = 0..v-1
        # This is itself a (Vandermonde) linear system in the v unknown magnitudes.
        mag_matrix = []
        for i in range(v):
            row = [gf256.gf_pow(gf256.gf_pow(gf256.GENERATOR, pos), i) for pos in error_positions]
            mag_matrix.append(row)
        mag_rhs = syndromes[:v]
        magnitudes = _solve_linear_system_gf256(mag_matrix, mag_rhs)
        if magnitudes is None:
            continue

        return error_positions, magnitudes

    return None


def rs_decode(received_symbols: list[int], n_parity: int) -> tuple[list[int], bool]:
    """
    Decode a received Reed-Solomon codeword.

    1. Compute syndromes; if all zero, no error occurred -- return the
       message symbols as-is (success=True).
    2. Otherwise, run PGZ to locate and correct up to
       MAX_CORRECTABLE_ERRORS symbol errors.
    3. If PGZ finds a consistent correction, apply it (XOR each error
       magnitude into its located position) and return the corrected
       message (success=True).
    4. If no consistent solution is found (too many errors for this
       implementation's documented scope), return the received data
       uncorrected with success=False, signalling "errors exceeded
       correction capacity" to the caller.

    Position indexing: `received_symbols` is treated with index 0 as
    the HIGHEST-degree coefficient (matches rs_encode's convention:
    received_symbols[0] is the first message symbol). PGZ's internal
    Chien search uses "distance from the low/constant-degree end", so
    positions are converted accordingly.
    """
    n_total = len(received_symbols)
    syndromes = _compute_syndromes(received_symbols, n_parity)

    if all(s == 0 for s in syndromes):
        return received_symbols[: n_total - n_parity], True

    result = _peterson_gorenstein_zierler(syndromes, n_total)
    if result is None:
        # Could not find a consistent correction within MAX_CORRECTABLE_ERRORS.
        return received_symbols[: n_total - n_parity], False

    error_positions, magnitudes = result
    corrected = received_symbols[:]
    for pos, mag in zip(error_positions, magnitudes):
        # pos is "distance from low/constant-degree end"; convert to a normal
        # list index (0 = highest degree = received_symbols[0]).
        list_index = n_total - 1 - pos
        if not (0 <= list_index < n_total):
            return received_symbols[: n_total - n_parity], False
        corrected[list_index] ^= mag

    # Verify the correction actually zeroes the syndromes (sanity check
    # against a spurious/incorrect PGZ solution before trusting it).
    verify_syndromes = _compute_syndromes(corrected, n_parity)
    if not all(s == 0 for s in verify_syndromes):
        return received_symbols[: n_total - n_parity], False

    return corrected[: n_total - n_parity], True


def rs_decode_report(received_symbols: list[int], n_parity: int) -> dict:
    syndromes = _compute_syndromes(received_symbols, n_parity)
    corrected, success = rs_decode(received_symbols, n_parity)
    return {
        "syndromes": syndromes,
        "success": success,
        "corrected_message": corrected,
        "max_correctable_errors": MAX_CORRECTABLE_ERRORS,
    }


def rs_encode_message(data: bytes, k: int = 16, n_parity: int = 4) -> bytes:
    """
    Apply rs_encode block-wise across an arbitrary-length byte stream:
    split `data` into blocks of `k` message symbols (zero-padding the
    final block if needed), append `n_parity` parity symbols to each
    block, and concatenate. This is the RS analogue of
    hamming_encode_message -- each block is coded independently, so a
    burst of bit errors confined to one block's bytes only costs that
    block some of its `MAX_CORRECTABLE_ERRORS` symbol-error budget,
    not the whole message's.
    """
    if k <= 0 or k > 255 - n_parity:
        raise ValueError("k must be positive and k + n_parity must not exceed 255 (GF(2^8) codeword limit)")
    padded = list(data) + [0] * ((-len(data)) % k)
    blocks = [padded[i:i + k] for i in range(0, len(padded), k)]
    encoded = bytearray()
    for block in blocks:
        encoded.extend(rs_encode(block, n_parity))
    return bytes(encoded)


def rs_decode_message(
    received: bytes, k: int = 16, n_parity: int = 4, original_length: int | None = None
) -> tuple[bytes, list[bool]]:
    """
    Inverse of rs_encode_message: split the received bytes into
    (k+n_parity)-symbol blocks, rs_decode each independently, and
    concatenate the recovered message symbols. Returns (decoded_bytes,
    per_block_success_flags). If `original_length` is given, trims the
    zero-padding added by rs_encode_message's final block.
    """
    block_len = k + n_parity
    if len(received) % block_len != 0:
        raise ValueError(f"received data length must be a multiple of {block_len}")
    decoded = bytearray()
    success_flags = []
    for i in range(0, len(received), block_len):
        block = list(received[i:i + block_len])
        corrected, success = rs_decode(block, n_parity)
        decoded.extend(corrected)
        success_flags.append(success)
    result = bytes(decoded)
    if original_length is not None:
        result = result[:original_length]
    return result, success_flags

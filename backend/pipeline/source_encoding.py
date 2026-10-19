"""Stage 1: Huffman source coding (from first principles; uses only ``heapq`` from the stdlib).

Mathematical background
-----------------------
For a source emitting symbols with probabilities p_i = f_i / N (f_i = frequency, N = total):

* Shannon entropy   H = - sum_i p_i * log2(p_i)   [bits/symbol]  -- the lower bound on the
  average length of ANY uniquely decodable binary code (source coding theorem).
* A *prefix code* assigns each symbol a bitstring such that no codeword is a prefix of
  another, so a bitstream can be decoded uniquely without separators.  Codeword lengths
  l_i of a binary prefix code satisfy the Kraft inequality  sum_i 2^(-l_i) <= 1.
* Average code length   L = sum_i p_i * l_i.   For Huffman codes  H <= L < H + 1.
* Efficiency   eta = H / L * 100 %   (defined only when L > 0).

Why Huffman's algorithm is optimal (exchange + induction argument)
------------------------------------------------------------------
Let a and b be the two least probable symbols.  (1) In some optimal prefix code a and b are
sibling leaves at maximum depth: if a more probable symbol were deeper than a less probable
one, swapping them would not increase L; the deepest leaf must have a sibling (otherwise its
codeword could be shortened); swapping that sibling with the other least-probable symbol
again cannot increase L.  (2) Merging a and b into one super-symbol of weight p_a + p_b
gives a smaller problem whose cost differs from the original by exactly p_a + p_b, a
constant independent of the rest of the tree.  So an optimal tree for the reduced problem
extends to an optimal tree for the original.  Induction on the alphabet size k gives
optimality of the greedy "merge the two lightest" rule.  Huffman is optimal among
*symbol-by-symbol binary prefix codes for the given probabilities*; it does not exploit
context, and it can still be improved on by e.g. block or arithmetic coding.

Conventions (all deterministic)
-------------------------------
* A "symbol" is one Unicode code point (one Python ``str`` character).  Combining marks and
  emoji sequences made of several code points are therefore several symbols.
* Tie-breaking: leaves are ordered by (frequency, code point) and numbered 0..k-1; every
  internal node gets the next number.  The heap key is (frequency, number).  The first node
  popped becomes the LEFT child (edge bit 0), the second the RIGHT child (edge bit 1).
  Consequently the code table depends only on the multiset of (symbol, count) pairs, not on
  the order in which symbols appear in the text.
* Empty input -> empty bitstring and empty code table.
* Single distinct symbol -> the tree is one leaf and that symbol gets the code "0"
  (length 1).  Entropy is 0 but L = 1, so the efficiency is 0 %: a zero-entropy source
  has no information, yet this code spends one bit per symbol so that the bitstream length
  still encodes the message length.
* The encoded bit length does NOT include the cost of transmitting the code table.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class HuffmanNode:
    """Node of a Huffman tree. Leaves carry a symbol; internal nodes carry two children."""
    freq: int
    order: int
    symbol: Optional[str] = None
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# --------------------------------------------------------------------------- validation
def _validate_freq(freq: dict[str, int], *, allow_empty: bool) -> None:
    if not isinstance(freq, dict):
        raise TypeError("frequency table must be a dict")
    if not freq and not allow_empty:
        raise ValueError("frequency table is empty")
    for sym, count in freq.items():
        if not isinstance(sym, str) or len(sym) != 1:
            raise ValueError(f"symbols must be single characters, got {sym!r}")
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"count for {sym!r} must be an int, got {type(count).__name__}")
        if count <= 0:
            raise ValueError(f"count for {sym!r} must be positive, got {count}")


def _validate_code_table(code_table: dict[str, str]) -> None:
    """Codes must be non-empty 0/1 strings, unique, and prefix-free."""
    if not isinstance(code_table, dict):
        raise TypeError("code table must be a dict")
    for sym, code in code_table.items():
        if not isinstance(sym, str) or len(sym) != 1:
            raise ValueError(f"code-table symbols must be single characters, got {sym!r}")
        if not isinstance(code, str) or not code or set(code) - {"0", "1"}:
            raise ValueError(f"code for {sym!r} must be a non-empty string of 0/1, got {code!r}")
    codes = sorted(code_table.values())
    for earlier, later in zip(codes, codes[1:]):
        if earlier == later:
            raise ValueError(f"duplicate codeword {earlier!r}")
        if later.startswith(earlier):          # sorted order puts a prefix right before its extensions
            raise ValueError(f"code table is not prefix-free: {earlier!r} is a prefix of {later!r}")


# --------------------------------------------------------------------------- core algorithm
def build_frequency_table(text: str) -> dict[str, int]:
    """Count occurrences of each Unicode code point, in order of first appearance."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def build_huffman_tree(freq: dict[str, int]) -> HuffmanNode:
    """Build the Huffman tree with a min-heap keyed by (frequency, creation order)."""
    _validate_freq(freq, allow_empty=False)
    leaves = sorted(freq.items(), key=lambda kv: (kv[1], ord(kv[0])))
    heap: list[tuple[int, int, HuffmanNode]] = []
    for order, (sym, count) in enumerate(leaves):
        heap.append((count, order, HuffmanNode(freq=count, order=order, symbol=sym)))
    heapq.heapify(heap)
    next_order = len(leaves)
    while len(heap) > 1:
        f1, _, first = heapq.heappop(heap)
        f2, _, second = heapq.heappop(heap)
        parent = HuffmanNode(freq=f1 + f2, order=next_order, left=first, right=second)
        heapq.heappush(heap, (parent.freq, next_order, parent))
        next_order += 1
    return heap[0][2]


def generate_codes(tree: HuffmanNode) -> dict[str, str]:
    """Read codewords off the tree (left edge = '0', right edge = '1'). Iterative DFS."""
    if not isinstance(tree, HuffmanNode):
        raise TypeError("tree must be a HuffmanNode")
    if tree.is_leaf:                        # single-symbol convention
        return {tree.symbol: "0"}
    codes: dict[str, str] = {}
    stack: list[tuple[HuffmanNode, str]] = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        if node.is_leaf:
            codes[node.symbol] = prefix
        else:
            stack.append((node.right, prefix + "1"))
            stack.append((node.left, prefix + "0"))
    return codes


def huffman_encode(text: str) -> tuple[str, dict[str, str]]:
    """Encode ``text``; returns (bitstring of '0'/'1' characters, code table)."""
    freq = build_frequency_table(text)
    if not freq:
        return "", {}
    codes = generate_codes(build_huffman_tree(freq))
    return "".join(codes[ch] for ch in text), codes


def huffman_decode(bits: str, code_table: dict[str, str]) -> str:
    """Decode a bitstring with a prefix-free code table.

    Raises ValueError for non-binary input, an invalid table, a bit sequence that matches
    no codeword, or a trailing incomplete codeword.
    """
    if not isinstance(bits, str):
        raise TypeError("bits must be a str")
    if set(bits) - {"0", "1"}:
        raise ValueError("bitstring may contain only the characters '0' and '1'")
    _validate_code_table(code_table)
    if not bits:
        return ""
    if not code_table:
        raise ValueError("cannot decode a non-empty bitstring with an empty code table")

    by_code = {code: sym for sym, code in code_table.items()}
    proper_prefixes = {code[:i] for code in by_code for i in range(1, len(code))}
    out: list[str] = []
    buffer = ""
    for bit in bits:
        buffer += bit
        if buffer in by_code:
            out.append(by_code[buffer])
            buffer = ""
        elif buffer not in proper_prefixes:
            raise ValueError(f"bit sequence {buffer!r} does not match any codeword")
    if buffer:
        raise ValueError(f"bitstring ends with an incomplete codeword {buffer!r}")
    return "".join(out)


# --------------------------------------------------------------------------- statistics
def calculate_entropy(freq: dict[str, int]) -> float:
    """Shannon entropy H = -sum p_i log2 p_i in bits/symbol (0.0 for an empty table)."""
    _validate_freq(freq, allow_empty=True)
    total = sum(freq.values())
    h = 0.0
    for count in freq.values():
        p = count / total
        h -= p * math.log2(p)
    return max(h, 0.0)


def calculate_average_code_length(freq: dict[str, int], code_table: dict[str, str]) -> float:
    """L = sum p_i * len(code_i) in bits/symbol (0.0 for an empty table)."""
    _validate_freq(freq, allow_empty=True)
    total = sum(freq.values())
    if total == 0:
        return 0.0
    length = 0.0
    for sym, count in freq.items():
        if sym not in code_table:
            raise ValueError(f"symbol {sym!r} has no codeword")
        length += (count / total) * len(code_table[sym])
    return length


def calculate_efficiency_percent(entropy: float, avg_code_length: float) -> Optional[float]:
    """eta = H / L * 100; ``None`` when L <= 0 (undefined, e.g. empty input)."""
    if avg_code_length <= 0:
        return None
    return entropy / avg_code_length * 100.0


# --------------------------------------------------------------------------- visualisation data
def tree_to_dict(root: HuffmanNode) -> dict[str, Any]:
    """JSON-compatible nested dict of the tree (iterative; left edge = 0, right edge = 1)."""
    def make(node: HuffmanNode) -> dict[str, Any]:
        return {"symbol": node.symbol, "freq": node.freq, "left": None, "right": None}

    top = make(root)
    stack = [(root, top)]
    while stack:
        node, data = stack.pop()
        if node.left is not None:
            data["left"] = make(node.left)
            stack.append((node.left, data["left"]))
        if node.right is not None:
            data["right"] = make(node.right)
            stack.append((node.right, data["right"]))
    return top

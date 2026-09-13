"""
source_encoding.py — Huffman coding (Information Theory).

Mathematical background
------------------------
Shannon's source coding theorem says the minimum possible average
number of bits per symbol needed to encode a source with symbol
probabilities p(x) is the *entropy*:

    H(X) = - sum_x p(x) * log2(p(x))     (bits/symbol)

Huffman coding is a greedy algorithm that constructs an optimal
*prefix-free* code (no codeword is a prefix of another, so a bitstream
can be decoded unambiguously left to right with no delimiters) whose
average code length L = sum_x p(x)*len(code(x)) satisfies
    H(X) <= L < H(X) + 1
i.e. Huffman gets within 1 bit of the theoretical entropy limit, and
exactly meets it when all probabilities are exact powers of 1/2.

Algorithm: repeatedly merge the two least-frequent nodes into a new
parent node (frequency = sum of the two), until one tree remains. This
is implemented with a min-heap for O(n log n) performance.
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=True)
class HuffmanNode:
    freq: int
    # tie-breaker counter ensures heapq never tries to compare characters/subtrees directly
    order: int = field(compare=True)
    char: Optional[str] = field(default=None, compare=False)
    left: Optional["HuffmanNode"] = field(default=None, compare=False)
    right: Optional["HuffmanNode"] = field(default=None, compare=False)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def build_frequency_table(text: str) -> dict[str, int]:
    """Count occurrences of each character in `text`."""
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def build_huffman_tree(freq: dict[str, int]) -> HuffmanNode:
    """
    Build the Huffman tree via the classic greedy min-heap merge:
    repeatedly pop the two lowest-frequency nodes and merge them under
    a new internal node whose frequency is their sum, until a single
    root remains. This greedy strategy is provably optimal (it
    produces a minimum weighted external path length tree) by an
    exchange argument: the two rarest symbols can always be made
    siblings at the deepest level of an optimal tree without
    increasing total cost.
    """
    if not freq:
        raise ValueError("cannot build a Huffman tree from empty input")

    counter = 0
    heap: list[HuffmanNode] = []
    for ch, f in freq.items():
        heapq.heappush(heap, HuffmanNode(freq=f, order=counter, char=ch))
        counter += 1

    if len(heap) == 1:
        # Special case: only one distinct symbol. Give it a trivial parent
        # so it still gets a 1-bit code ("0") rather than an empty string.
        only = heap[0]
        parent = HuffmanNode(freq=only.freq, order=counter, left=only, right=None)
        return parent

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent = HuffmanNode(freq=a.freq + b.freq, order=counter, left=a, right=b)
        counter += 1
        heapq.heappush(heap, parent)

    return heap[0]


def generate_codes(tree: HuffmanNode) -> dict[str, str]:
    """
    Walk the tree assigning '0' for every left branch and '1' for every
    right branch; a symbol's code is the bit-path from root to its leaf.
    Because every symbol lives at a leaf, and no leaf is an ancestor of
    another, the resulting code is automatically prefix-free.
    """
    codes: dict[str, str] = {}

    def walk(node: Optional[HuffmanNode], path: str):
        if node is None:
            return
        if node.is_leaf():
            codes[node.char] = path if path else "0"
            return
        walk(node.left, path + "0")
        walk(node.right, path + "1")

    walk(tree, "")
    return codes


def huffman_encode(text: str) -> tuple[str, dict[str, str]]:
    """
    Full source-encoding pipeline: build frequency table -> build tree
    -> derive codes -> concatenate each character's code into one
    bitstring. Returns (bitstring, code_table) so the receiver can
    reconstruct the same prefix tree implicitly via the table.
    """
    if text == "":
        return "", {}
    freq = build_frequency_table(text)
    tree = build_huffman_tree(freq)
    codes = generate_codes(tree)
    bitstring = "".join(codes[ch] for ch in text)
    return bitstring, codes


def huffman_decode(bits: str, code_table: dict[str, str]) -> str:
    """
    Decode by walking the bitstring and, at each position, matching the
    longest prefix that appears in the (inverted) code table -- since
    the code is prefix-free this greedy left-to-right match is always
    unambiguous and never needs backtracking.
    """
    if bits == "":
        return ""
    reverse_table = {code: ch for ch, code in code_table.items()}
    result = []
    current = ""
    for bit in bits:
        current += bit
        if current in reverse_table:
            result.append(reverse_table[current])
            current = ""
    if current != "":
        raise ValueError("leftover bits do not form a valid code -- corrupted stream or table")
    return "".join(result)


def shannon_entropy(freq: dict[str, int]) -> float:
    """
    H(X) = -sum p(x) log2 p(x), the theoretical minimum average bits
    per symbol for this source's empirical symbol distribution.
    """
    total = sum(freq.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def average_code_length(freq: dict[str, int], codes: dict[str, str]) -> float:
    """L = sum p(x) * len(code(x)) -- the actual achieved average code length."""
    total = sum(freq.values())
    if total == 0:
        return 0.0
    return sum(count * len(codes[ch]) for ch, count in freq.items()) / total


def huffman_encode_report(text: str) -> dict:
    """Convenience wrapper producing the full API response payload for this stage."""
    if text == "":
        return {
            "bitstring": "",
            "code_table": {},
            "entropy": 0.0,
            "avg_code_length": 0.0,
            "original_bits": 0,
            "encoded_bits": 0,
        }
    freq = build_frequency_table(text)
    bitstring, codes = huffman_encode(text)
    entropy = shannon_entropy(freq)
    avg_len = average_code_length(freq, codes)
    return {
        "bitstring": bitstring,
        "code_table": codes,
        "entropy": round(entropy, 4),
        "avg_code_length": round(avg_len, 4),
        "original_bits": len(text) * 8,  # naive fixed-width baseline (UTF-8-ish, 8 bits/char)
        "encoded_bits": len(bitstring),
    }

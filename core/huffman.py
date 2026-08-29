"""
core/huffman.py

Huffman Compression and its inverse, built from first principles: a manual
priority queue over tree nodes, manual merging, manual code-table
generation from tree traversal.

The stage operates on the OUTPUT of BinaryEncodingStage (a bit string) by
regrouping it back into byte-sized symbols (0-255) to build frequency
statistics -- this keeps the "everything flows as bits between stages"
contract while still compressing at the byte level, which is where the
redundancy actually lives for text.

Forward stage : bit string -> {"encoded_bits", "code_table", "num_symbols"}
Inverse stage : that dict   -> bit string

Math concept demonstrated: information theory (Shannon entropy), greedy
tree construction.
"""

import heapq
import itertools
import math

from core.pipeline import Stage, Trace


class HuffmanNode:
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


def bits_to_bytes(bits: str) -> list[int]:
    return [int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)]


def bytes_to_bits(byte_list: list[int]) -> str:
    return "".join(format(b, "08b") for b in byte_list)


def build_frequency_table(symbols: list[int]) -> dict[int, int]:
    freq: dict[int, int] = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    return freq


def shannon_entropy(freq: dict[int, int], total: int) -> float:
    """Theoretical minimum average bits/symbol for this distribution."""
    entropy = 0.0
    for f in freq.values():
        p = f / total
        entropy -= p * math.log2(p)
    return entropy


def build_huffman_tree(freq: dict[int, int], trace: Trace | None = None) -> HuffmanNode:
    """
    Classic greedy construction: repeatedly pop the two lowest-frequency
    nodes and merge them, until one tree remains.
    A counter breaks ties so heapq never has to compare HuffmanNode objects
    directly (which have no natural ordering).
    """
    counter = itertools.count()
    heap = []
    for symbol, f in freq.items():
        heapq.heappush(heap, (f, next(counter), HuffmanNode(f, symbol=symbol)))

    if len(heap) == 1:
        # Single distinct symbol: still need a real branch so its code
        # isn't the empty string.
        f, _, node = heap[0]
        return HuffmanNode(f, left=node)

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = HuffmanNode(f1 + f2, left=n1, right=n2)
        if trace is not None:
            trace.log_step(
                action="merge_nodes",
                left_symbol=n1.symbol, left_freq=f1,
                right_symbol=n2.symbol, right_freq=f2,
                new_freq=merged.freq,
            )
        heapq.heappush(heap, (merged.freq, next(counter), merged))

    return heap[0][2]


def build_code_table(root: HuffmanNode) -> dict[int, str]:
    codes: dict[int, str] = {}

    def walk(node: HuffmanNode, prefix: str):
        if node.is_leaf():
            codes[node.symbol] = prefix or "0"
            return
        if node.left:
            walk(node.left, prefix + "0")
        if node.right:
            walk(node.right, prefix + "1")

    walk(root, "")
    return codes


class HuffmanCompressionStage(Stage):
    name = "Huffman Compression"

    def run(self, data: str, trace: Trace) -> dict:
        symbols = bits_to_bytes(data)
        total = len(symbols)
        freq = build_frequency_table(symbols)
        entropy = shannon_entropy(freq, total) if total else 0.0

        if total == 0:
            # Empty input: no symbols, nothing to build a tree from.
            # Handle explicitly rather than letting build_huffman_tree
            # index into an empty heap.
            trace.log_step(action="empty_input", note="no symbols to compress")
            trace.metadata["distinct_symbols"] = 0
            trace.metadata["entropy_bits_per_symbol"] = 0.0
            trace.metadata["theoretical_min_bits"] = 0.0
            trace.metadata["original_bits"] = len(data)
            trace.metadata["compressed_bits"] = 0
            trace.metadata["compression_ratio"] = 1.0
            return {"encoded_bits": "", "code_table": {}, "num_symbols": 0}

        tree = build_huffman_tree(freq, trace)
        codes = build_code_table(tree)
        encoded_bits = "".join(codes[s] for s in symbols)

        trace.metadata["distinct_symbols"] = len(freq)
        trace.metadata["entropy_bits_per_symbol"] = entropy
        trace.metadata["theoretical_min_bits"] = entropy * total
        trace.metadata["original_bits"] = len(data)
        trace.metadata["compressed_bits"] = len(encoded_bits)
        trace.metadata["compression_ratio"] = (
            len(data) / len(encoded_bits) if encoded_bits else 1.0
        )

        # JSON-friendly keys (symbols must become strings for the trace/API layer)
        code_table = {str(sym): code for sym, code in codes.items()}
        return {"encoded_bits": encoded_bits, "code_table": code_table, "num_symbols": total}


class HuffmanDecompressionStage(Stage):
    name = "Huffman Decompression"

    def run(self, data: dict, trace: Trace) -> str:
        encoded_bits = data["encoded_bits"]
        code_table = data["code_table"]
        num_symbols = data["num_symbols"]

        reverse_table = {code: int(sym) for sym, code in code_table.items()}

        symbols: list[int] = []
        buffer = ""
        for bit in encoded_bits:
            buffer += bit
            if buffer in reverse_table:
                symbol = reverse_table[buffer]
                symbols.append(symbol)
                trace.log_step(action="decode_symbol", code=buffer, symbol=symbol)
                buffer = ""
                if len(symbols) == num_symbols:
                    break

        return bytes_to_bits(symbols)

"""
core/encoding.py

Source Encoding / Binary Representation and its inverse.

Forward stage : text (str)  -> bit string (str of '0'/'1'), via UTF-8 bytes
Inverse stage : bit string  -> text (str)

Math concept demonstrated: information representation, byte/bit structure.
"""

from core.pipeline import Stage, Trace


def text_to_bits(text: str) -> str:
    """UTF-8 encode text, then render every byte as an 8-bit binary string."""
    return "".join(format(b, "08b") for b in text.encode("utf-8"))


def bits_to_text(bits: str) -> str:
    """Inverse of text_to_bits: group into bytes, decode as UTF-8."""
    byte_vals = [int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)]
    return bytes(byte_vals).decode("utf-8", errors="replace")


class BinaryEncodingStage(Stage):
    name = "Source / Binary Encoding"

    def run(self, data: str, trace: Trace) -> str:
        byte_data = data.encode("utf-8")
        bits = ""
        for b in byte_data:
            byte_bits = format(b, "08b")
            trace.log_step(action="encode_byte", byte=b, bits=byte_bits)
            bits += byte_bits
        trace.metadata["num_bytes"] = len(byte_data)
        trace.metadata["num_bits"] = len(bits)
        return bits


class BinaryDecodingStage(Stage):
    name = "Binary / Source Decoding"

    def run(self, data: str, trace: Trace) -> str:
        byte_vals = [int(data[i:i + 8], 2) for i in range(0, len(data), 8)]
        trace.log_step(action="decode_bytes", num_bytes=len(byte_vals))
        return bytes(byte_vals).decode("utf-8", errors="replace")

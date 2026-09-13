"""
models.py — Pydantic v2 request/response schemas for every pipeline endpoint.

Kept in one module (rather than one file per stage) since these are thin
data-transfer objects with no independent logic of their own -- all the
actual math lives in backend/pipeline/*.py.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# 1. Source encoding (Huffman)
# ---------------------------------------------------------------------------
class HuffmanEncodeRequest(BaseModel):
    text: str = Field(..., description="Plaintext message to Huffman-encode")


class HuffmanEncodeResponse(BaseModel):
    bitstring: str
    code_table: dict[str, str]
    entropy: float
    avg_code_length: float
    original_bits: int
    encoded_bits: int


class HuffmanDecodeRequest(BaseModel):
    bits: str
    code_table: dict[str, str]


class HuffmanDecodeResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# 2. Binary representation
# ---------------------------------------------------------------------------
class BinaryEncodeRequest(BaseModel):
    text: str


class BinaryEncodeResponse(BaseModel):
    bits: str
    byte_grid: list[dict]


class BinaryDecodeRequest(BaseModel):
    bits: str


class BinaryDecodeResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# 3. Compression (LZ77)
# ---------------------------------------------------------------------------
class CompressRequest(BaseModel):
    data_hex: str = Field(..., description="Input data to compress, as a hex string")
    window_size: int = 4096
    lookahead_size: int = 18


class LZ77Token(BaseModel):
    offset: int
    length: int
    next_byte: int


class CompressResponse(BaseModel):
    tokens: list[LZ77Token]
    num_tokens: int
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: Optional[float]


# ---------------------------------------------------------------------------
# 4. Encryption (AES / RSA)
# ---------------------------------------------------------------------------
class AESEncryptRequest(BaseModel):
    plaintext_hex: str
    key_hex: Optional[str] = Field(None, description="16-byte hex AES key; random if omitted")
    iv_hex: Optional[str] = Field(None, description="16-byte hex IV; random if omitted")


class AESEncryptResponse(BaseModel):
    ciphertext_hex: str
    key_hex: str
    iv_hex: str
    round_keys: list[str]


class AESDecryptRequest(BaseModel):
    ciphertext_hex: str
    key_hex: str
    iv_hex: str


class AESDecryptResponse(BaseModel):
    plaintext_hex: str
    plaintext_text: Optional[str] = None


class RSAKeyExchangeRequest(BaseModel):
    aes_key_hex: str
    bits: int = 32


class RSAKeyExchangeResponse(BaseModel):
    p: int
    q: int
    n: int
    phi: int
    e: int
    d: int
    wrapped_key_int: int


class RSAUnwrapRequest(BaseModel):
    wrapped_key_int: int
    d: int
    n: int
    key_length_bytes: int = 16


class RSAUnwrapResponse(BaseModel):
    aes_key_hex: str


# ---------------------------------------------------------------------------
# 5. Packetization
# ---------------------------------------------------------------------------
class PacketizeRequest(BaseModel):
    data_hex: str
    payload_size: int = 64


class PacketModel(BaseModel):
    seq_num: int
    total_packets: int
    payload_hex: str
    payload_size: int
    header_checksum: int


class PacketizeResponse(BaseModel):
    packets: list[PacketModel]
    num_packets: int


# ---------------------------------------------------------------------------
# 6. Error detection (CRC-32)
# ---------------------------------------------------------------------------
class CRCRequest(BaseModel):
    data_hex: str


class CRCResponse(BaseModel):
    checksum: int
    checksum_hex: str


class CRCVerifyRequest(BaseModel):
    data_hex: str
    checksum: int


class CRCVerifyResponse(BaseModel):
    valid: bool


# ---------------------------------------------------------------------------
# 7. Routing (Dijkstra)
# ---------------------------------------------------------------------------
class TopologyResponse(BaseModel):
    nodes: list[str]
    edges: list[dict]


class EdgeRequest(BaseModel):
    source: str
    target: str
    weight: float


class ShortestPathRequest(BaseModel):
    source: str
    target: str
    edges: Optional[list[EdgeRequest]] = Field(
        None, description="Custom topology; default topology used if omitted"
    )


class ShortestPathResponse(BaseModel):
    path: list[str]
    total_cost: Optional[float]
    relaxation_steps: list[dict]
    reachable: bool


# ---------------------------------------------------------------------------
# 8. Channel transmission
# ---------------------------------------------------------------------------
class ChannelTransmitRequest(BaseModel):
    bits: str
    flip_probability: float = Field(0.05, ge=0.0, le=1.0)
    seed: Optional[int] = None
    channel_type: Literal["bsc", "awgn"] = "bsc"
    sigma: float = 1.0  # used only for awgn


class ChannelTransmitResponse(BaseModel):
    received_bits: str
    num_flips: int
    positions_flipped: list[int]
    ber_observed: float
    ber_theoretical: float


# ---------------------------------------------------------------------------
# 9. Error correction (Hamming / Reed-Solomon)
# ---------------------------------------------------------------------------
class HammingEncodeRequest(BaseModel):
    data_bits: str


class HammingEncodeResponse(BaseModel):
    codeword_bits: str


class HammingDecodeRequest(BaseModel):
    received_bits: str


class HammingDecodeResponse(BaseModel):
    syndrome: int
    error_position: Optional[int]
    corrected_codeword: str
    corrected_data_bits: str


class ReedSolomonEncodeRequest(BaseModel):
    data_hex: str
    k: int = 16
    n_parity: int = 4


class ReedSolomonEncodeResponse(BaseModel):
    encoded_hex: str


class ReedSolomonDecodeRequest(BaseModel):
    received_hex: str
    k: int = 16
    n_parity: int = 4
    original_length: Optional[int] = None


class ReedSolomonDecodeResponse(BaseModel):
    decoded_hex: str
    block_success_flags: list[bool]
    max_correctable_errors: int


# ---------------------------------------------------------------------------
# 10 & 11. Decryption / Decoding
# ---------------------------------------------------------------------------
class DecodeFullRequest(BaseModel):
    tokens: list[LZ77Token]
    code_table: dict[str, str]
    original_text: str


class DecodeFullResponse(BaseModel):
    success: bool
    original: str
    recovered: str
    diff_positions: list[int]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class PipelineRunRequest(BaseModel):
    message: str = Field(..., min_length=1)
    aes_key_bits: Literal[128] = 128
    noise_probability: float = Field(0.02, ge=0.0, le=0.3)
    error_correction_scheme: Literal["hamming", "reed-solomon"] = "hamming"
    topology_source: str = "A"
    topology_target: str = "F"
    channel_type: Literal["bsc", "awgn"] = "bsc"
    seed: Optional[int] = None


class PipelineStageTrace(BaseModel):
    stage_name: str
    input_snapshot: dict
    output_snapshot: dict
    math_detail: str


class PipelineRunResponse(BaseModel):
    success: bool
    original_message: str
    recovered_message: Optional[str]
    diff_positions: list[int]
    failure_reason: Optional[str] = None
    trace: list[PipelineStageTrace]

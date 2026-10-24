"""Pydantic v2 request/response models for the API.

Size limits are deliberately modest: this is an interactive teaching tool, not a bulk
processor. Limits are enforced twice: by field constraints here (HTTP 422) and by a global
request-body cap in ``main.py`` (HTTP 413).
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from backend.pipeline.compression import (
    DEFAULT_LOOKAHEAD_SIZE, DEFAULT_WINDOW_SIZE, MAX_LOOKAHEAD_SIZE, MAX_WINDOW_SIZE,
)

MAX_TEXT_CHARS = 10_000          # characters accepted by the text-based endpoints
MAX_BODY_BYTES = 1_000_000       # global request-body cap (Content-Length)
MAX_LZ77_BYTES = 40_000          # bytes accepted by the LZ77 endpoint (= 10,000 chars x 4 UTF-8 bytes)
MAX_TOKENS_RETURNED = 2_000      # per-token records returned (statistics always cover ALL tokens)
MAX_AES_BYTES = 40_000           # plaintext bytes accepted by the AES endpoint
MAX_AES_BLOCKS_RETURNED = 64     # per-block records returned (the ciphertext is always complete)
MAX_RSA_MODULUS_BITS = 512       # largest modulus the API will generate
MAX_RSA_PRIME_BITS = 256         # largest user-supplied prime (p, q)
MAX_RSA_STEPS_RETURNED = 64      # square-and-multiply / Euclid steps returned per computation
DEFAULT_RSA_WRAP_BITS = 160      # default modulus when carrying a 128-bit AES key (n must exceed 2^128)
DEFAULT_RSA_INTEGER_BITS = 32    # default tiny modulus for the integer demonstration
MIN_RSA_WRAP_BITS = 130          # smallest generated modulus guaranteed to exceed any 128-bit key
MAX_DECIMAL_DIGITS = 78          # decimal strings for p, q, e (<= 259 bits)
MAX_PACKETIZE_BYTES = 40_000     # bytes accepted by the packetization endpoint
MAX_API_PAYLOAD_SIZE = 4096      # largest payload_size the API accepts (the module allows 65535)
MAX_PACKETS_RETURNED = 256       # per-packet records returned per table (statistics cover ALL packets)
MAX_PAYLOAD_PREVIEW_BYTES = 128  # payload bytes returned per packet record
MAX_EXPERIMENT_SEQUENCES = 1000  # entries accepted in each drop/duplicate/corrupt list
MAX_CRC_BYTES = 40_000           # bytes accepted by the CRC endpoint
MAX_CRC_TRACE_BYTES = 256        # bytes of per-byte register steps returned
MAX_CRC_BIT_TRACE_BYTES = 8      # bytes of bit-level shift/XOR steps returned
MAX_BURST_BITS = 4096            # longest burst the experiment can inject
MAX_FLIPPED_BITS_LISTED = 1000   # net flipped bit positions listed in a response
MAX_MESSAGE_DIGITS = 160         # decimal string for the integer message


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- shared
class TextRequest(StrictModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS,
                      description="Unicode text. Must be encodable as UTF-8.")

    # Defence in depth: with the pinned Pydantic version, pydantic-core already rejects lone
    # surrogates ('string_unicode') before this validator runs. It is kept (and unit-tested
    # directly) so the guarantee does not silently depend on that library behaviour.
    @field_validator("text")
    @classmethod
    def _must_be_utf8_encodable(cls, v: str) -> str:
        try:
            v.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("text contains characters that cannot be encoded as UTF-8 "
                             "(for example a lone surrogate)")
        return v


# --------------------------------------------------------------------------- Stage 1: Huffman
class HuffmanRequest(TextRequest):
    pass


class SymbolStat(BaseModel):
    symbol: str
    code_point: str = Field(description="Unicode code point, e.g. 'U+0041'")
    count: int
    probability: float
    code: str
    code_length: int


class HuffmanResponse(BaseModel):
    bitstring: str
    code_table: dict[str, str]
    entropy: float
    avg_code_length: float
    efficiency_percent: Optional[float] = Field(
        description="H / L * 100; null when undefined (empty input, L = 0)")
    # --- additional detail used by the visualisation
    symbols: list[SymbolStat]
    tree: Optional[dict[str, Any]]
    original_length_chars: int
    original_length_bits_utf8: int
    encoded_bit_length: int
    ratio_vs_utf8: Optional[float] = Field(
        description="utf8 bits / encoded bits; null when the encoded length is 0")
    table_overhead_included: bool = Field(
        description="Always false: the code table must also be transmitted for decoding.")
    round_trip_ok: bool = Field(description="Server decoded the bitstring and compared it to the input.")
    single_symbol_convention: bool = Field(
        description="True when the single distinct symbol was assigned the code '0'.")


# --------------------------------------------------------------------------- Stage 2: binary
class BinaryRequest(TextRequest):
    pass


class ByteInfo(BaseModel):
    index: int
    binary: str
    bits: list[int]
    hex: str
    decimal: int
    role: str
    character: Optional[str]
    char_index: Optional[int]
    char_start: bool


class CharacterInfo(BaseModel):
    index: int
    character: str
    code_point: int
    code_point_hex: str
    utf8_hex: str
    utf8_byte_count: int


class BinaryResponse(BaseModel):
    bitstring: str
    byte_count: int
    character_count: int
    bytes: list[ByteInfo]
    characters: list[CharacterInfo]
    round_trip_ok: bool


# --------------------------------------------------------------------------- Stage 3: LZ77
class LZ77Request(StrictModel):
    """Exactly one of ``text`` (UTF-8 encoded) or ``data_hex`` (raw bytes as hex) is required."""
    text: Optional[str] = Field(None, max_length=MAX_TEXT_CHARS)
    data_hex: Optional[str] = Field(None, max_length=2 * MAX_LZ77_BYTES,
                                    description="Raw bytes as an even-length hex string.")
    window_size: StrictInt = Field(DEFAULT_WINDOW_SIZE, ge=1, le=MAX_WINDOW_SIZE)
    lookahead_size: StrictInt = Field(DEFAULT_LOOKAHEAD_SIZE, ge=1, le=MAX_LOOKAHEAD_SIZE)

    @field_validator("text")
    @classmethod
    def _text_is_utf8_encodable(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                v.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("text contains characters that cannot be encoded as UTF-8")
        return v

    @field_validator("data_hex")
    @classmethod
    def _hex_is_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if len(v) % 2:
                raise ValueError("data_hex must have an even number of hex digits")
            if set(v) - set("0123456789abcdefABCDEF"):
                raise ValueError("data_hex may contain only the characters 0-9, a-f, A-F")
        return v

    @model_validator(mode="after")
    def _exactly_one_input(self) -> "LZ77Request":
        if (self.text is None) == (self.data_hex is None):
            raise ValueError("provide exactly one of 'text' or 'data_hex'")
        return self


class LZ77TokenInfo(BaseModel):
    index: int
    position: int
    kind: str = Field(description="literal | match | match-end (end-of-input token, empty next byte)")
    offset: int
    length: int
    next_hex: str
    next_char: Optional[str]
    covers: int
    match_source_start: Optional[int]
    match_hex: str
    overlapping: bool
    window_start: int
    lookahead_end: int
    bits: int


class LZ77Response(BaseModel):
    window_size: int
    lookahead_size: int
    original_bytes: int
    token_count: int
    literal_count: int
    match_count: int
    offset_bits: int
    length_bits: int
    payload_bits: int
    header_bytes: int
    compressed_bytes: int
    compression_ratio: Optional[float] = Field(description="original / compressed; null if undefined")
    space_saved_percent: Optional[float] = Field(
        description="(1 - compressed/original) * 100; negative means expansion; null for empty input")
    expanded: bool = Field(description="True when the serialised form is larger than the input")
    tokens: list[LZ77TokenInfo]
    tokens_returned: int
    tokens_truncated: bool
    round_trip_ok: bool = Field(description="Server decompressed the tokens and compared with the input bytes.")
    serialization_round_trip_ok: bool = Field(
        description="serialize -> deserialize reproduced the tokens, and the serialised size matches the statistics.")
    size_note: str


# --------------------------------------------------------------------------- Stage 4B / 11: AES
_HEX_CHARS = set("0123456789abcdefABCDEF")


def _check_hex(v: str, name: str) -> str:
    if len(v) % 2:
        raise ValueError(f"{name} must have an even number of hex digits")
    if set(v) - _HEX_CHARS:
        raise ValueError(f"{name} may contain only the characters 0-9, a-f, A-F")
    return v


class AESEncryptRequest(StrictModel):
    """Plaintext is exactly one of ``text`` or ``data_hex``. Key/IV are optional: when omitted
    they are generated (from ``seed`` if given, which is reproducible but NOT secure)."""
    text: Optional[str] = Field(None, max_length=MAX_TEXT_CHARS)
    data_hex: Optional[str] = Field(None, max_length=2 * MAX_AES_BYTES)
    key_hex: Optional[str] = Field(None, min_length=32, max_length=32, description="16-byte key as 32 hex digits")
    iv_hex: Optional[str] = Field(None, min_length=32, max_length=32, description="16-byte IV as 32 hex digits")
    seed: Optional[StrictInt] = Field(None, ge=0, le=2 ** 32 - 1)
    key_bits: StrictInt = 128
    include_trace: StrictBool = True
    reveal_round_keys: StrictBool = Field(
        True, description="The trace needs the round keys to show AddRoundKey, and round key 0 IS the key. "
                          "Set false to redact all round-key material from the trace.")
    trace_block: StrictInt = Field(0, ge=0, le=MAX_AES_BYTES // 16 + 1)

    @field_validator("text")
    @classmethod
    def _text_is_utf8_encodable(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                v.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("text contains characters that cannot be encoded as UTF-8")
        return v

    @field_validator("data_hex")
    @classmethod
    def _data_hex_valid(cls, v: Optional[str]) -> Optional[str]:
        return _check_hex(v, "data_hex") if v is not None else v

    @field_validator("key_hex")
    @classmethod
    def _key_hex_valid(cls, v: Optional[str]) -> Optional[str]:
        return _check_hex(v, "key_hex") if v is not None else v

    @field_validator("iv_hex")
    @classmethod
    def _iv_hex_valid(cls, v: Optional[str]) -> Optional[str]:
        return _check_hex(v, "iv_hex") if v is not None else v

    @field_validator("key_bits")
    @classmethod
    def _key_bits_known(cls, v: int) -> int:
        if v not in (128, 192, 256):
            raise ValueError("key_bits must be 128, 192 or 256")
        return v

    @model_validator(mode="after")
    def _exactly_one_plaintext(self) -> "AESEncryptRequest":
        if (self.text is None) == (self.data_hex is None):
            raise ValueError("provide exactly one of 'text' or 'data_hex'")
        return self


class AESBlockInfo(BaseModel):
    index: int
    plaintext_hex: str = Field(description="Padded plaintext block P_i")
    cipher_input_hex: str = Field(description="P_i XOR C_(i-1) (C_0 = IV): the actual input to the block cipher")
    ciphertext_hex: str = Field(description="C_i")


class AESEncryptResponse(BaseModel):
    algorithm: str
    mode: str
    padding: str
    key_bits: int
    key_hex: Optional[str] = Field(description="Returned ONLY when the server generated the key; never when you supplied it. "
                                              "(A trace with reveal_round_keys=true still contains round key 0, which equals the key.)")
    key_generated: bool
    key_seeded: bool = Field(description="True if key/IV came from a reproducible seed (insecure; experiments only).")
    iv_hex: str
    iv_generated: bool
    plaintext_bytes: int
    padding_bytes: int
    padded_bytes: int
    block_count: int
    ciphertext_hex: str
    blocks: list[AESBlockInfo]
    blocks_truncated: bool
    trace: Optional[dict[str, Any]] = Field(description="State after every transformation for block `trace_block`. Contains round keys: secret material.")
    trace_block: Optional[int]
    round_trip_ok: bool = Field(description="Server decrypted the ciphertext and compared it with the input bytes.")
    warning: str


class AESDecryptRequest(StrictModel):
    ciphertext_hex: str = Field(..., max_length=2 * (MAX_AES_BYTES + 16))
    key_hex: str = Field(..., min_length=32, max_length=32)
    iv_hex: str = Field(..., min_length=32, max_length=32)
    include_trace: StrictBool = True
    reveal_round_keys: StrictBool = True
    trace_block: StrictInt = Field(0, ge=0, le=MAX_AES_BYTES // 16 + 1)

    @field_validator("ciphertext_hex")
    @classmethod
    def _ct_hex_valid(cls, v: str) -> str:
        return _check_hex(v, "ciphertext_hex")

    @field_validator("key_hex")
    @classmethod
    def _key_hex_valid(cls, v: str) -> str:
        return _check_hex(v, "key_hex")

    @field_validator("iv_hex")
    @classmethod
    def _iv_hex_valid(cls, v: str) -> str:
        return _check_hex(v, "iv_hex")


class AESDecryptBlockInfo(BaseModel):
    index: int
    ciphertext_hex: str
    decrypted_hex: str = Field(description="D_K(C_i) before the CBC XOR")
    plaintext_hex: str = Field(description="D_K(C_i) XOR C_(i-1), still padded")


class AESDecryptResponse(BaseModel):
    algorithm: str
    mode: str
    padding: str
    plaintext_hex: str
    plaintext_bytes: int
    plaintext_text: Optional[str] = Field(description="UTF-8 decoding of the plaintext, or null if it is not valid UTF-8.")
    plaintext_is_valid_utf8: bool
    padding_bytes: int
    block_count: int
    blocks: list[AESDecryptBlockInfo]
    blocks_truncated: bool
    trace: Optional[dict[str, Any]]
    trace_block: Optional[int]
    warning: str


# --------------------------------------------------------------------------- Stage 5: RSA key exchange
_DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)$")


def _check_decimal(v: str, name: str) -> str:
    if not _DECIMAL_RE.match(v):
        raise ValueError(f"{name} must be a non-negative integer written in decimal digits "
                         f"(no sign, spaces, separators or leading zeros)")
    return v


class RSAKeyExchangeRequest(StrictModel):
    """Educational RSA demonstration. All large integers are exchanged as DECIMAL STRINGS because
    JavaScript cannot represent integers above 2^53 exactly.

    ``mode`` ``aes_key_wrap``: encrypt a 128-bit AES key (integer < n) with the receiver's public key.
    ``mode`` ``integer``: encrypt a small integer message < n (tiny keys; NOT real key transport).
    Supply either ``modulus_bits`` (generate keys) or both ``p`` and ``q`` (inspect your own primes).
    """
    mode: str = "aes_key_wrap"
    modulus_bits: Optional[StrictInt] = Field(None, ge=12, le=MAX_RSA_MODULUS_BITS,
                                              description="Even; default 160 (aes_key_wrap) or 32 (integer).")
    p: Optional[str] = Field(None, max_length=MAX_DECIMAL_DIGITS)
    q: Optional[str] = Field(None, max_length=MAX_DECIMAL_DIGITS)
    e: Optional[str] = Field(None, max_length=MAX_DECIMAL_DIGITS)
    message: Optional[str] = Field(None, max_length=MAX_MESSAGE_DIGITS)
    aes_key_hex: Optional[str] = Field(None, min_length=32, max_length=32)
    seed: Optional[StrictInt] = Field(None, ge=0, le=2 ** 32 - 1)
    include_traces: StrictBool = Field(
        True, description="Square-and-multiply traces contain the message integer (for aes_key_wrap: the key). "
                          "Set false to omit them.")

    @field_validator("mode")
    @classmethod
    def _mode_known(cls, v: str) -> str:
        if v not in ("aes_key_wrap", "integer"):
            raise ValueError("mode must be 'aes_key_wrap' or 'integer'")
        return v

    @field_validator("modulus_bits")
    @classmethod
    def _bits_even(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v % 2:
            raise ValueError("modulus_bits must be even (n is the product of two primes of equal size)")
        return v

    @field_validator("p")
    @classmethod
    def _p_decimal(cls, v: Optional[str]) -> Optional[str]:
        return _check_decimal(v, "p") if v is not None else v

    @field_validator("q")
    @classmethod
    def _q_decimal(cls, v: Optional[str]) -> Optional[str]:
        return _check_decimal(v, "q") if v is not None else v

    @field_validator("e")
    @classmethod
    def _e_decimal(cls, v: Optional[str]) -> Optional[str]:
        return _check_decimal(v, "e") if v is not None else v

    @field_validator("message")
    @classmethod
    def _message_decimal(cls, v: Optional[str]) -> Optional[str]:
        return _check_decimal(v, "message") if v is not None else v

    @field_validator("aes_key_hex")
    @classmethod
    def _key_hex(cls, v: Optional[str]) -> Optional[str]:
        return _check_hex(v, "aes_key_hex") if v is not None else v

    @model_validator(mode="after")
    def _consistent_fields(self) -> "RSAKeyExchangeRequest":
        if (self.p is None) != (self.q is None):
            raise ValueError("provide both 'p' and 'q', or neither")
        if self.p is not None and self.modulus_bits is not None:
            raise ValueError("'modulus_bits' cannot be combined with explicit 'p' and 'q'")
        if self.mode == "integer" and self.aes_key_hex is not None:
            raise ValueError("'aes_key_hex' is only valid with mode 'aes_key_wrap'")
        if self.mode == "aes_key_wrap" and self.message is not None:
            raise ValueError("'message' is only valid with mode 'integer'")
        return self


class RSAKeyExchangeResponse(BaseModel):
    mode: str
    insecure: bool = Field(description="Always true: educational textbook RSA with tiny keys.")
    security_note: str
    real_world_minimum_modulus_bits: int
    textbook_notes: list[str]
    roles: dict[str, str]
    key_source: str = Field(description="'generated' or 'user_primes'")
    seeded: bool
    modulus_bits: int
    p: str
    q: str
    n: str
    phi: str
    e: str
    d: str
    e_source: str = Field(description="'default' (65537), 'fallback' (smaller standard exponent) or 'user'")
    key_checks: dict[str, Any]
    primality: dict[str, Any]
    d_derivation: dict[str, Any]
    message_kind: str
    message: Optional[str] = Field(description="The integer being encrypted (null for a supplied AES key).")
    message_bits: int
    aes_key_hex: Optional[str] = Field(description="Only when the server generated the demo AES key.")
    recovered_key_hex: Optional[str] = Field(description="Only when the server generated the demo AES key.")
    ciphertext: str
    recovered: Optional[str] = Field(description="Decrypted integer as a decimal string; null when a SUPPLIED aes_key_hex is being carried.")
    encryption: Optional[dict[str, Any]]
    decryption: Optional[dict[str, Any]]
    verification: dict[str, Any]
    round_trip_ok: bool
    key_recovered_ok: Optional[bool]


# --------------------------------------------------------------------------- Stage 6: packetization
SequenceNumber = Annotated[int, Field(strict=True, ge=0)]


class PacketizeRequest(StrictModel):
    """Split data into packets, then optionally run a controlled transmission experiment on them.

    Experiment order (deterministic): corrupt headers -> duplicate -> drop (explicit, then random loss) -> shuffle.
    Sequence numbers in the three lists refer to the ORIGINAL packet numbering (0 .. packet_count-1).
    """
    text: Optional[str] = Field(None, max_length=MAX_TEXT_CHARS)
    data_hex: Optional[str] = Field(None, max_length=2 * MAX_PACKETIZE_BYTES)
    payload_size: StrictInt = Field(64, ge=1, le=MAX_API_PAYLOAD_SIZE)
    loss_probability: float = Field(0.0, ge=0, le=1, strict=True)
    seed: Optional[StrictInt] = Field(None, ge=0, le=2 ** 32 - 1)
    drop_sequences: list[SequenceNumber] = Field(default_factory=list, max_length=MAX_EXPERIMENT_SEQUENCES)
    duplicate_sequences: list[SequenceNumber] = Field(default_factory=list, max_length=MAX_EXPERIMENT_SEQUENCES)
    corrupt_header_sequences: list[SequenceNumber] = Field(default_factory=list, max_length=MAX_EXPERIMENT_SEQUENCES)
    shuffle: StrictBool = False
    allow_identical_duplicates: StrictBool = Field(
        False, description="Tolerate repeated packets whose payload equals the first copy (conflicting copies never are).")

    @field_validator("text")
    @classmethod
    def _text_is_utf8_encodable(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                v.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("text contains characters that cannot be encoded as UTF-8")
        return v

    @field_validator("data_hex")
    @classmethod
    def _data_hex_valid(cls, v: Optional[str]) -> Optional[str]:
        return _check_hex(v, "data_hex") if v is not None else v

    @model_validator(mode="after")
    def _exactly_one_input(self) -> "PacketizeRequest":
        if (self.text is None) == (self.data_hex is None):
            raise ValueError("provide exactly one of 'text' or 'data_hex'")
        return self


class PacketInfo(BaseModel):
    position: int = Field(description="Arrival position in this table (0-based)")
    sequence: int
    total: int
    payload_length: int
    serialized_size: int = Field(description="header + payload bytes on the wire (not the payload size)")
    header_hex: str
    checksum: int
    checksum_hex: str
    checksum_valid: bool
    status: str = Field(description="ok | corrupted_header | duplicate")
    payload_hex: str = Field(description="First 128 payload bytes")
    payload_truncated: bool


class PacketizeResponse(BaseModel):
    data_bytes: int
    payload_size: int
    header_size: int
    stats: dict[str, Any]
    packets: list[PacketInfo]
    packets_returned: int
    packets_truncated: bool
    lossless_round_trip_ok: bool = Field(description="Reassembling the untouched packets reproduced the input exactly.")
    experiment: dict[str, Any]
    received_packets: list[PacketInfo]
    received_truncated: bool
    diagnosis: dict[str, Any]
    reassembly: dict[str, Any]
    header_checksum_note: str


# --------------------------------------------------------------------------- Stage 7: CRC-32
class CRCRequest(StrictModel):
    """Compute (and optionally verify) CRC-32, then optionally corrupt the data on purpose.

    Bit positions count in TRANSMISSION order: bit p is bit (p mod 8) of byte (p div 8), bit 0 being the least
    significant bit of the byte. Positions in ``flip_bits`` and the ``burst`` are XOR-ed into the data, so a bit
    named twice (or by both) is flipped back and ends up unchanged.
    """
    text: Optional[str] = Field(None, max_length=MAX_TEXT_CHARS)
    data_hex: Optional[str] = Field(None, max_length=2 * MAX_CRC_BYTES)
    expected_crc_hex: Optional[str] = Field(None, min_length=8, max_length=8,
                                            description="8 hex digits; if given the data is verified against it")
    flip_bits: list[SequenceNumber] = Field(default_factory=list, max_length=MAX_EXPERIMENT_SEQUENCES)
    burst_start_bit: Optional[StrictInt] = Field(None, ge=0)
    burst_length: Optional[StrictInt] = Field(None, ge=1, le=MAX_BURST_BITS,
                                              description="Flips every bit from burst_start_bit for burst_length bits")
    forge_to_match_original: StrictBool = Field(
        False, description="After the corruption, append 4 computed bytes so the CRC equals the ORIGINAL data's CRC.")
    trace_bytes: StrictInt = Field(16, ge=0, le=MAX_CRC_TRACE_BYTES)
    bit_trace_bytes: StrictInt = Field(2, ge=0, le=MAX_CRC_BIT_TRACE_BYTES)
    include_table: StrictBool = True

    @field_validator("text")
    @classmethod
    def _text_is_utf8_encodable(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                v.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("text contains characters that cannot be encoded as UTF-8")
        return v

    @field_validator("data_hex")
    @classmethod
    def _data_hex_valid(cls, v: Optional[str]) -> Optional[str]:
        return _check_hex(v, "data_hex") if v is not None else v

    @field_validator("expected_crc_hex")
    @classmethod
    def _expected_hex_valid(cls, v: Optional[str]) -> Optional[str]:
        return _check_hex(v, "expected_crc_hex") if v is not None else v

    @model_validator(mode="after")
    def _consistent_fields(self) -> "CRCRequest":
        if (self.text is None) == (self.data_hex is None):
            raise ValueError("provide exactly one of 'text' or 'data_hex'")
        if (self.burst_start_bit is None) != (self.burst_length is None):
            raise ValueError("provide both 'burst_start_bit' and 'burst_length', or neither")
        if self.forge_to_match_original and not (self.flip_bits or self.burst_length is not None):
            raise ValueError("'forge_to_match_original' needs a corruption ('flip_bits' or a burst) to repair")
        return self


class CRCResponse(BaseModel):
    data_bytes: int
    crc32: int
    crc32_hex: str
    crc32_little_endian_hex: str = Field(description="The 4 bytes a sender appends")
    parameters: dict[str, Any]
    polynomial: dict[str, Any]
    cross_checks: dict[str, Any]
    standard_check_value_ok: bool = Field(description="crc32('123456789') == 0xCBF43926, recomputed on every request")
    residue_check: bool = Field(description="crc32(data + crc32 little-endian) == 0x2144DF1C")
    verification: Optional[dict[str, Any]]
    trace: dict[str, Any]
    table_hex: Optional[list[str]]
    corruption: Optional[dict[str, Any]]
    forged: Optional[dict[str, Any]]
    properties: list[str]

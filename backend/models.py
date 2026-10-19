"""Pydantic v2 request/response models for the API.

Size limits are deliberately modest: this is an interactive teaching tool, not a bulk
processor. Limits are enforced twice: by field constraints here (HTTP 422) and by a global
request-body cap in ``main.py`` (HTTP 413).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from backend.pipeline.compression import (
    DEFAULT_LOOKAHEAD_SIZE, DEFAULT_WINDOW_SIZE, MAX_LOOKAHEAD_SIZE, MAX_WINDOW_SIZE,
)

MAX_TEXT_CHARS = 10_000          # characters accepted by the text-based endpoints
MAX_BODY_BYTES = 1_000_000       # global request-body cap (Content-Length)
MAX_LZ77_BYTES = 40_000          # bytes accepted by the LZ77 endpoint (= 10,000 chars x 4 UTF-8 bytes)
MAX_TOKENS_RETURNED = 2_000      # per-token records returned (statistics always cover ALL tokens)


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

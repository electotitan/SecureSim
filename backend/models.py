"""Pydantic v2 request/response models for the API.

Size limits are deliberately modest: this is an interactive teaching tool, not a bulk
processor. Limits are enforced twice: by field constraints here (HTTP 422) and by a global
request-body cap in ``main.py`` (HTTP 413).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TEXT_CHARS = 10_000          # characters accepted by the text-based endpoints
MAX_BODY_BYTES = 1_000_000       # global request-body cap (Content-Length)


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

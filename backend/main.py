"""FastAPI application entry point.

Checkpoint 1 scope: application scaffold only.
  * GET /            -> serves the frontend (frontend/index.html)
  * GET /api/health  -> health check
  * /static/*        -> frontend static assets (CSS, JS, components)
  * /docs            -> interactive API documentation (FastAPI default)
  * /api/docs        -> alias of the interactive documentation

Stage endpoints are added checkpoint by checkpoint:
  * POST /api/encode/huffman  (Checkpoint 3)
  * POST /api/encode/binary   (Checkpoint 3)
  * POST /api/compress/lz77   (Checkpoint 4)
  * POST /api/encrypt/aes     (Checkpoint 5)
  * POST /api/decrypt/aes     (Checkpoint 5)
"""
from __future__ import annotations

import random
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html

from backend.errors import IntegrityFailure, ValidationFailure, error_body, register_error_handlers
from backend.models import (
    MAX_AES_BLOCKS_RETURNED, MAX_BODY_BYTES, MAX_TOKENS_RETURNED, AESDecryptRequest,
    AESDecryptResponse, AESEncryptRequest, AESEncryptResponse, BinaryRequest, BinaryResponse,
    HuffmanRequest, HuffmanResponse, LZ77Request, LZ77Response,
)
from backend.pipeline import (
    binary_repr, compression, decryption_aes, encryption_aes, source_encoding,
)

APP_NAME = "Interactive Mathematical Simulator for Secure Digital Communication"
APP_VERSION = "0.5.0"

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Educational simulator. All algorithms are implemented from first "
        "principles and are NOT suitable for protecting real sensitive data."
    ),
)


register_error_handlers(app)


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Reject oversized bodies early (declared Content-Length). Field-level max_length
    constraints in models.py additionally bound what is processed."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content=error_body("validation_error",
                               f"Request body too large (limit {MAX_BODY_BYTES} bytes)."))
    return await call_next(request)


@app.get("/api/health", tags=["system"])
def health() -> dict:
    """Health check used by the frontend connection indicator."""
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}


@app.post("/api/encode/huffman", response_model=HuffmanResponse, tags=["stage 1: source coding"])
def encode_huffman(req: HuffmanRequest) -> HuffmanResponse:
    """Huffman-encode text and report entropy, average code length and efficiency."""
    try:
        text = req.text
        freq = source_encoding.build_frequency_table(text)
        bits, codes = source_encoding.huffman_encode(text)
        total = len(text)
        entropy = source_encoding.calculate_entropy(freq)
        avg_len = source_encoding.calculate_average_code_length(freq, codes)
        efficiency = source_encoding.calculate_efficiency_percent(entropy, avg_len)
        tree = source_encoding.tree_to_dict(source_encoding.build_huffman_tree(freq)) if freq else None
        symbols = [
            {"symbol": sym, "code_point": f"U+{ord(sym):04X}", "count": count,
             "probability": count / total, "code": codes[sym], "code_length": len(codes[sym])}
            for sym, count in sorted(freq.items(), key=lambda kv: (-kv[1], ord(kv[0])))
        ]
        utf8_bits = 8 * len(text.encode("utf-8"))
        # Verify the result by actually decoding it (never assume success).
        round_trip_ok = source_encoding.huffman_decode(bits, codes) == text
    except ValueError as exc:
        raise ValidationFailure(str(exc)) from exc
    return HuffmanResponse(
        bitstring=bits, code_table=codes, entropy=entropy, avg_code_length=avg_len,
        efficiency_percent=efficiency, symbols=symbols, tree=tree,
        original_length_chars=total, original_length_bits_utf8=utf8_bits,
        encoded_bit_length=len(bits),
        ratio_vs_utf8=(utf8_bits / len(bits)) if bits else None,
        table_overhead_included=False, round_trip_ok=round_trip_ok,
        single_symbol_convention=len(freq) == 1,
    )


@app.post("/api/encode/binary", response_model=BinaryResponse, tags=["stage 2: binary representation"])
def encode_binary(req: BinaryRequest) -> BinaryResponse:
    """UTF-8 encode text into a bitstring with a per-byte breakdown."""
    try:
        bits = binary_repr.text_to_binary(req.text)
        grid = binary_repr.visualize_bits(bits)
        characters = binary_repr.describe_characters(req.text)
        round_trip_ok = binary_repr.binary_to_text(bits) == req.text
    except ValueError as exc:
        raise ValidationFailure(str(exc)) from exc
    return BinaryResponse(
        bitstring=bits, byte_count=len(grid), character_count=len(req.text),
        bytes=grid, characters=characters, round_trip_ok=round_trip_ok,
    )


@app.post("/api/compress/lz77", response_model=LZ77Response, tags=["stage 3: compression"])
def compress_lz77(req: LZ77Request) -> LZ77Response:
    """LZ77-compress text (UTF-8) or raw bytes; sizes come from a concrete serialisation."""
    w, l = req.window_size, req.lookahead_size
    try:
        data = req.text.encode("utf-8") if req.text is not None else bytes.fromhex(req.data_hex)
        tokens = compression.lz77_compress(data, w, l)
        stats = compression.compression_stats(len(data), tokens, w, l)
        described = compression.describe_tokens(data, tokens, w, l, limit=MAX_TOKENS_RETURNED)
        # Verify by actually reversing the transformation (never assume success).
        round_trip_ok = compression.lz77_decompress(tokens, window_size=w, lookahead_size=l) == data
    except ValueError as exc:
        raise ValidationFailure(str(exc)) from exc
    # Self-check of the size statistics. A failure here is a server-side inconsistency, not a
    # user error, so it is reported as a flag instead of being turned into a validation error.
    try:
        blob = compression.serialize_tokens(tokens, w, l)
        serialization_ok = (compression.deserialize_tokens(blob, w, l) == tokens
                            and len(blob) == stats["compressed_bytes"])
    except ValueError:
        serialization_ok = False
    return LZ77Response(
        window_size=w, lookahead_size=l, **stats, tokens=described,
        tokens_returned=len(described), tokens_truncated=len(described) < len(tokens),
        round_trip_ok=round_trip_ok, serialization_round_trip_ok=serialization_ok,
        size_note=("compressed_bytes is the size of the fixed-width serialisation (1 header byte + "
                   "packed tokens), not the token count; each token costs offset_bits + length_bits "
                   "+ 1 + 8 bits (the last 8 only if a next byte is present), so data without "
                   "repeats can expand."),
    )


AES_WARNING = ("Educational AES-128-CBC implementation: not constant-time, no authentication, and padding "
               "errors are distinguishable. Never use it to protect real data.")


def _derive_key_iv(seed):
    """Key then IV are ALWAYS drawn in this order so a seed reproduces both regardless of what
    the caller supplied. With a seed the source is random.Random (reproducible, NOT secure)."""
    if seed is None:
        return secrets.token_bytes(16), secrets.token_bytes(16)
    rng = random.Random(seed)
    return rng.randbytes(16), rng.randbytes(16)


def _redact_round_keys(trace: dict) -> dict:
    """Remove all round-key material from a trace (round key 0 equals the cipher key)."""
    trace["round_keys_hex"] = None
    for step in trace["steps"]:
        step.pop("round_key_hex", None)
    return trace


def _padding_length(plain_padded: bytes) -> int:
    return plain_padded[-1] if plain_padded else 0


@app.post("/api/encrypt/aes", response_model=AESEncryptResponse, tags=["stage 4: AES"])
def encrypt_aes(req: AESEncryptRequest) -> AESEncryptResponse:
    """AES-128-CBC encryption (PKCS#7) with an optional per-transformation trace of one block."""
    if req.key_bits != 128:
        raise HTTPException(status_code=501,
                            detail="Only AES-128 is implemented; AES-192 and AES-256 are not supported.")
    data = req.text.encode("utf-8") if req.text is not None else bytes.fromhex(req.data_hex)
    gen_key, gen_iv = _derive_key_iv(req.seed)
    key = bytes.fromhex(req.key_hex) if req.key_hex is not None else gen_key
    iv = bytes.fromhex(req.iv_hex) if req.iv_hex is not None else gen_iv
    try:
        records = encryption_aes.aes_cbc_encrypt_blocks(data, key, iv)
        ciphertext = b"".join(r["ciphertext"] for r in records)
        trace = None
        if req.include_trace:
            if req.trace_block >= len(records):
                raise ValidationFailure(
                    f"trace_block {req.trace_block} is out of range (message has {len(records)} blocks)",
                    [{"field": "body.trace_block", "message": "out of range", "code": "out_of_range"}])
            trace = encryption_aes.aes_encrypt_block_trace(records[req.trace_block]["cipher_input"], key)
            if not req.reveal_round_keys:
                trace = _redact_round_keys(trace)
        try:      # verify by actually decrypting (never assume success)
            round_trip_ok = decryption_aes.aes_cbc_decrypt(ciphertext, key, iv) == data
        except encryption_aes.AESError:
            round_trip_ok = False
    except ValueError as exc:
        if isinstance(exc, encryption_aes.AESError):
            raise
        raise ValidationFailure(str(exc)) from exc
    shown = records[:MAX_AES_BLOCKS_RETURNED]
    key_generated = req.key_hex is None
    return AESEncryptResponse(
        algorithm="AES-128", mode="CBC", padding="PKCS#7", key_bits=128,
        key_hex=key.hex() if key_generated else None, key_generated=key_generated,
        key_seeded=req.seed is not None and (key_generated or req.iv_hex is None),
        iv_hex=iv.hex(), iv_generated=req.iv_hex is None,
        plaintext_bytes=len(data), padding_bytes=len(records) * 16 - len(data),
        padded_bytes=len(records) * 16, block_count=len(records), ciphertext_hex=ciphertext.hex(),
        blocks=[{"index": i, "plaintext_hex": r["plaintext"].hex(), "cipher_input_hex": r["cipher_input"].hex(),
                 "ciphertext_hex": r["ciphertext"].hex()} for i, r in enumerate(shown)],
        blocks_truncated=len(shown) < len(records),
        trace=trace, trace_block=req.trace_block if trace is not None else None,
        round_trip_ok=round_trip_ok, warning=AES_WARNING,
    )


@app.post("/api/decrypt/aes", response_model=AESDecryptResponse, tags=["stage 11: AES decryption"])
def decrypt_aes(req: AESDecryptRequest) -> AESDecryptResponse:
    """AES-128-CBC decryption with PKCS#7 validation. Bad length/padding -> integrity_failure."""
    ciphertext, key, iv = bytes.fromhex(req.ciphertext_hex), bytes.fromhex(req.key_hex), bytes.fromhex(req.iv_hex)
    try:
        records = decryption_aes.aes_cbc_decrypt_blocks(ciphertext, key, iv)
        padded = b"".join(r["plaintext"] for r in records)
        plaintext = decryption_aes.pkcs7_unpad(padded)
        trace = None
        if req.include_trace:
            if req.trace_block >= len(records):
                raise ValidationFailure(
                    f"trace_block {req.trace_block} is out of range (ciphertext has {len(records)} blocks)",
                    [{"field": "body.trace_block", "message": "out of range", "code": "out_of_range"}])
            trace = decryption_aes.aes_decrypt_block_trace(records[req.trace_block]["ciphertext"], key)
            if not req.reveal_round_keys:
                trace = _redact_round_keys(trace)
    except encryption_aes.AESError as exc:
        raise IntegrityFailure(f"Decryption failed: {exc}") from exc
    try:
        text, utf8_ok = plaintext.decode("utf-8"), True
    except UnicodeDecodeError:
        text, utf8_ok = None, False
    shown = records[:MAX_AES_BLOCKS_RETURNED]
    return AESDecryptResponse(
        algorithm="AES-128", mode="CBC", padding="PKCS#7",
        plaintext_hex=plaintext.hex(), plaintext_bytes=len(plaintext),
        plaintext_text=text, plaintext_is_valid_utf8=utf8_ok,
        padding_bytes=_padding_length(padded), block_count=len(records),
        blocks=[{"index": i, "ciphertext_hex": r["ciphertext"].hex(), "decrypted_hex": r["decrypted"].hex(),
                 "plaintext_hex": r["plaintext"].hex()} for i, r in enumerate(shown)],
        blocks_truncated=len(shown) < len(records),
        trace=trace, trace_block=req.trace_block if trace is not None else None, warning=AES_WARNING,
    )


@app.get("/api/docs", include_in_schema=False, response_class=HTMLResponse)
def api_docs() -> HTMLResponse:
    """Alias for the interactive documentation (spec lists /api/docs)."""
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{APP_NAME} - API docs")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Static assets: /static/style.css, /static/app.js, /static/components/*.js
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

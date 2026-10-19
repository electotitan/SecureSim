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
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html

from backend.errors import ValidationFailure, error_body, register_error_handlers
from backend.models import (
    MAX_BODY_BYTES, BinaryRequest, BinaryResponse, HuffmanRequest, HuffmanResponse,
)
from backend.pipeline import binary_repr, source_encoding

APP_NAME = "Interactive Mathematical Simulator for Secure Digital Communication"
APP_VERSION = "0.3.0"

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

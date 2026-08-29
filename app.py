"""
app.py

API layer wrapping the real Phase 2 pipeline for the frontend/visualizer.

A fresh RSA keypair is generated per request by default (so every run is
self-contained and reproducible from its own trace). The channel noise
probability is exposed as a request parameter so the frontend can let
users experiment with it directly -- e.g. sweeping p from 0 to 0.1 and
watching Hamming correction succeed, then start to fail, exactly as in
the BER-sweep test in tests/test_phase2_modules.py.

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload
Then POST to http://127.0.0.1:8000/run with JSON:
    {"message": "hello", "flip_probability": 0.02, "key_bits": 16}
or open http://127.0.0.1:8000/ for the (still bare -- Phase 4 builds the
real visualizer) debug frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.pipeline import build_default_pipeline

app = FastAPI(title="Secure Digital Communication Simulator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    message: str
    flip_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    key_bits: int = Field(default=16, ge=8, le=64)


@app.post("/run")
def run_pipeline(req: RunRequest) -> dict:
    """
    Runs the message through the full 14-stage pipeline (binary encoding,
    Huffman compression, RSA encryption, packetization, Hamming encoding,
    CRC attach, Dijkstra routing, noisy channel, CRC check, Hamming
    correction, reassembly, RSA decryption, Huffman decompression, binary
    decoding) and returns every stage's trace, ready for the visualizer.
    """
    pipeline = build_default_pipeline(
        channel_flip_probability=req.flip_probability,
        key_bits=req.key_bits,
    )
    return pipeline.run(req.message)


@app.get("/")
def serve_debug_frontend():
    return FileResponse("viz/index.html")


app.mount("/static", StaticFiles(directory="viz"), name="static")

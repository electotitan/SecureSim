"""
main.py — FastAPI application wiring every pipeline stage endpoint plus
the full-pipeline orchestrator, and serving the static frontend.

Run with:  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations
import secrets

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pathlib

from . import models as m
from .orchestrator import run_pipeline
from .pipeline import source_encoding as se
from .pipeline import binary_repr as br
from .pipeline import compression as lz
from .pipeline import encryption_aes as aes
from .pipeline import decryption_aes as aes_dec
from .pipeline import encryption_rsa as rsa
from .pipeline import packetization as pkt
from .pipeline import error_detection_crc as crc
from .pipeline import routing_dijkstra as rd
from .pipeline import channel as ch
from .pipeline import error_correction_hamming as hm
from .pipeline import error_correction_rs as rs
from .pipeline import decoding as dec

app = FastAPI(
    title="Interactive Mathematical Simulator for Secure Digital Communication",
    description="Every stage of a secure digital communication pipeline, implemented from first principles.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 1. Source Encoding (Huffman)
# ---------------------------------------------------------------------------
@app.post("/api/encode/huffman", response_model=m.HuffmanEncodeResponse)
def encode_huffman(req: m.HuffmanEncodeRequest):
    report = se.huffman_encode_report(req.text)
    return report


@app.post("/api/decode/huffman", response_model=m.HuffmanDecodeResponse)
def decode_huffman(req: m.HuffmanDecodeRequest):
    try:
        text = se.huffman_decode(req.bits, req.code_table)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return m.HuffmanDecodeResponse(text=text)


# ---------------------------------------------------------------------------
# 2. Binary Representation
# ---------------------------------------------------------------------------
@app.post("/api/encode/binary", response_model=m.BinaryEncodeResponse)
def encode_binary(req: m.BinaryEncodeRequest):
    bits = br.text_to_binary(req.text)
    grid = br.visualize_bits(bits)
    return m.BinaryEncodeResponse(bits=bits, byte_grid=grid)


@app.post("/api/decode/binary", response_model=m.BinaryDecodeResponse)
def decode_binary(req: m.BinaryDecodeRequest):
    try:
        text = br.binary_to_text(req.bits)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, str(exc))
    return m.BinaryDecodeResponse(text=text)


# ---------------------------------------------------------------------------
# 3. Compression (LZ77)
# ---------------------------------------------------------------------------
@app.post("/api/compress/lz77", response_model=m.CompressResponse)
def compress_lz77(req: m.CompressRequest):
    try:
        data = bytes.fromhex(req.data_hex)
    except ValueError:
        raise HTTPException(400, "data_hex is not valid hex")
    report = lz.compression_report(data, req.window_size, req.lookahead_size)
    return report


# ---------------------------------------------------------------------------
# 4. Encryption (AES / RSA)
# ---------------------------------------------------------------------------
@app.post("/api/encrypt/aes", response_model=m.AESEncryptResponse)
def encrypt_aes(req: m.AESEncryptRequest):
    try:
        plaintext = bytes.fromhex(req.plaintext_hex)
    except ValueError:
        raise HTTPException(400, "plaintext_hex is not valid hex")
    key = bytes.fromhex(req.key_hex) if req.key_hex else secrets.token_bytes(16)
    iv = bytes.fromhex(req.iv_hex) if req.iv_hex else secrets.token_bytes(16)
    if len(key) != 16:
        raise HTTPException(400, "key must be exactly 16 bytes (AES-128)")
    ciphertext = aes.aes_cbc_encrypt(plaintext, key, iv)
    round_keys = aes.key_expansion(key)
    return m.AESEncryptResponse(
        ciphertext_hex=ciphertext.hex(),
        key_hex=key.hex(),
        iv_hex=iv.hex(),
        round_keys=[rk.hex() for rk in round_keys],
    )


@app.post("/api/decrypt/aes", response_model=m.AESDecryptResponse)
def decrypt_aes(req: m.AESDecryptRequest):
    try:
        ciphertext = bytes.fromhex(req.ciphertext_hex)
        key = bytes.fromhex(req.key_hex)
        iv = bytes.fromhex(req.iv_hex)
        plaintext = aes_dec.aes_cbc_decrypt(ciphertext, key, iv)
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, f"decryption failed: {exc}")
    text = None
    try:
        text = plaintext.decode("utf-8")
    except UnicodeDecodeError:
        pass
    return m.AESDecryptResponse(plaintext_hex=plaintext.hex(), plaintext_text=text)


@app.post("/api/encrypt/rsa-keyexchange", response_model=m.RSAKeyExchangeResponse)
def rsa_keyexchange(req: m.RSAKeyExchangeRequest):
    try:
        aes_key = bytes.fromhex(req.aes_key_hex)
    except ValueError:
        raise HTTPException(400, "aes_key_hex is not valid hex")
    bits = req.bits
    for _ in range(6):
        keys = rsa.generate_keypair(bits=bits)
        try:
            wrapped = rsa.wrap_aes_key(aes_key, keys["public_key"])
            return m.RSAKeyExchangeResponse(
                p=keys["p"], q=keys["q"], n=keys["n"], phi=keys["phi"],
                e=keys["e"], d=keys["d"], wrapped_key_int=wrapped,
            )
        except ValueError:
            bits += 8
    raise HTTPException(400, "could not generate an RSA modulus large enough for this key length; try a larger `bits`")


@app.post("/api/decrypt/rsa-unwrap", response_model=m.RSAUnwrapResponse)
def rsa_unwrap(req: m.RSAUnwrapRequest):
    aes_key = rsa.unwrap_aes_key(req.wrapped_key_int, {"d": req.d, "n": req.n}, req.key_length_bytes)
    return m.RSAUnwrapResponse(aes_key_hex=aes_key.hex())


# ---------------------------------------------------------------------------
# 5. Packetization
# ---------------------------------------------------------------------------
@app.post("/api/packetize", response_model=m.PacketizeResponse)
def packetize(req: m.PacketizeRequest):
    try:
        data = bytes.fromhex(req.data_hex)
    except ValueError:
        raise HTTPException(400, "data_hex is not valid hex")
    report = pkt.packetize_report(data, req.payload_size)
    return report


# ---------------------------------------------------------------------------
# 6. Error Detection (CRC-32)
# ---------------------------------------------------------------------------
@app.post("/api/error-detect/crc", response_model=m.CRCResponse)
def error_detect_crc(req: m.CRCRequest):
    try:
        data = bytes.fromhex(req.data_hex)
    except ValueError:
        raise HTTPException(400, "data_hex is not valid hex")
    checksum = crc.crc32_compute(data)
    return m.CRCResponse(checksum=checksum, checksum_hex=format(checksum, "08x"))


@app.post("/api/error-detect/crc-verify", response_model=m.CRCVerifyResponse)
def error_detect_crc_verify(req: m.CRCVerifyRequest):
    try:
        data = bytes.fromhex(req.data_hex)
    except ValueError:
        raise HTTPException(400, "data_hex is not valid hex")
    return m.CRCVerifyResponse(valid=crc.crc32_verify(data, req.checksum))


# ---------------------------------------------------------------------------
# 7. Routing (Dijkstra)
# ---------------------------------------------------------------------------
@app.get("/api/routing/topology", response_model=m.TopologyResponse)
def routing_topology():
    topo = rd.default_topology()
    return topo.to_dict()


@app.post("/api/routing/shortest-path", response_model=m.ShortestPathResponse)
def routing_shortest_path(req: m.ShortestPathRequest):
    if req.edges:
        graph = rd.Graph()
        for e in req.edges:
            graph.add_edge(e.source, e.target, e.weight)
    else:
        graph = rd.default_topology()
    try:
        report = rd.shortest_path_report(graph, req.source, req.target)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return report


# ---------------------------------------------------------------------------
# 8. Channel Transmission
# ---------------------------------------------------------------------------
@app.post("/api/channel/transmit", response_model=m.ChannelTransmitResponse)
def channel_transmit(req: m.ChannelTransmitRequest):
    if any(b not in "01" for b in req.bits):
        raise HTTPException(400, "bits must be a string of '0'/'1' characters")
    if req.channel_type == "awgn":
        result = ch.awgn_transmit(req.bits, sigma=req.sigma, seed=req.seed)
    else:
        result = ch.bsc_transmit(req.bits, req.flip_probability, seed=req.seed)
    return result


# ---------------------------------------------------------------------------
# 9. Error Correction (Hamming / Reed-Solomon)
# ---------------------------------------------------------------------------
@app.post("/api/error-correct/hamming/encode", response_model=m.HammingEncodeResponse)
def hamming_encode_endpoint(req: m.HammingEncodeRequest):
    try:
        codeword = hm.hamming_encode_message(req.data_bits)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return m.HammingEncodeResponse(codeword_bits=codeword)


@app.post("/api/error-correct/hamming", response_model=m.HammingDecodeResponse)
def hamming_decode_endpoint(req: m.HammingDecodeRequest):
    try:
        result = hm.hamming_decode_detailed(req.received_bits)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return result


@app.post("/api/error-correct/reed-solomon/encode", response_model=m.ReedSolomonEncodeResponse)
def rs_encode_endpoint(req: m.ReedSolomonEncodeRequest):
    try:
        data = bytes.fromhex(req.data_hex)
        encoded = rs.rs_encode_message(data, k=req.k, n_parity=req.n_parity)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return m.ReedSolomonEncodeResponse(encoded_hex=encoded.hex())


@app.post("/api/error-correct/reed-solomon", response_model=m.ReedSolomonDecodeResponse)
def rs_decode_endpoint(req: m.ReedSolomonDecodeRequest):
    try:
        received = bytes.fromhex(req.received_hex)
        decoded, flags = rs.rs_decode_message(received, k=req.k, n_parity=req.n_parity, original_length=req.original_length)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return m.ReedSolomonDecodeResponse(
        decoded_hex=decoded.hex(), block_success_flags=flags, max_correctable_errors=rs.MAX_CORRECTABLE_ERRORS
    )


# ---------------------------------------------------------------------------
# 11. Decoding (final reverse pipeline + comparison)
# ---------------------------------------------------------------------------
@app.post("/api/decode/full", response_model=m.DecodeFullResponse)
def decode_full_endpoint(req: m.DecodeFullRequest):
    tokens = [(t.offset, t.length, t.next_byte) for t in req.tokens]
    try:
        result = dec.decode_full(tokens, req.code_table, req.original_text)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return result


# ---------------------------------------------------------------------------
# Orchestrator — runs all 11 stages end-to-end
# ---------------------------------------------------------------------------
@app.post("/api/pipeline/run", response_model=m.PipelineRunResponse)
def pipeline_run(req: m.PipelineRunRequest):
    try:
        result = run_pipeline(
            message=req.message,
            noise_probability=req.noise_probability,
            error_correction_scheme=req.error_correction_scheme,
            topology_source=req.topology_source,
            topology_target=req.topology_target,
            channel_type=req.channel_type,
            seed=req.seed,
        )
    except Exception as exc:
        raise HTTPException(500, f"pipeline execution failed: {type(exc).__name__}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
_FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent / "frontend"

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))

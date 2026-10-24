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
  * POST /api/encrypt/rsa-keyexchange (Checkpoint 6)
  * POST /api/packetize        (Checkpoint 7)
  * POST /api/error-detect/crc (Checkpoint 8)
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
    DEFAULT_RSA_INTEGER_BITS, DEFAULT_RSA_WRAP_BITS, MAX_AES_BLOCKS_RETURNED, MAX_BODY_BYTES,
    MAX_FLIPPED_BITS_LISTED, MAX_PACKETS_RETURNED, MAX_PAYLOAD_PREVIEW_BYTES, MAX_RSA_PRIME_BITS,
    MAX_RSA_STEPS_RETURNED, MAX_TOKENS_RETURNED, MIN_RSA_WRAP_BITS, CRCRequest, CRCResponse,
    PacketizeRequest, PacketizeResponse, AESDecryptRequest, AESDecryptResponse, AESEncryptRequest, AESEncryptResponse, BinaryRequest,
    BinaryResponse, HuffmanRequest, HuffmanResponse, LZ77Request, LZ77Response,
    RSAKeyExchangeRequest, RSAKeyExchangeResponse,
)
from backend.pipeline import (
    binary_repr, compression, decryption_aes, decryption_rsa, encryption_aes, encryption_rsa,
    error_detection_crc, packetization, source_encoding,
)

APP_NAME = "Interactive Mathematical Simulator for Secure Digital Communication"
APP_VERSION = "0.8.0"

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


RSA_TEXTBOOK_NOTES = [
    "Textbook RSA has no padding (no OAEP): encryption is deterministic, so equal messages give equal ciphertexts.",
    "It is malleable: E(a) * E(b) mod n = E(a*b mod n), so ciphertexts can be altered meaningfully without the key.",
    "The messages 0, 1 and n-1 encrypt to themselves; raw RSA only accepts integers smaller than the modulus.",
    "These moduli are tiny and can be factored almost instantly; real RSA needs at least 2048 bits and a vetted library.",
    "A 128-bit AES key as an integer is below 2^128, so the modulus must exceed 2^128 to carry it (at least 129 bits).",
]
_RSA_FIELD_MAP = {"bits": "modulus_bits"}


def _s(value) -> str:
    """Big integers cross the JSON boundary as decimal strings (JavaScript numbers are exact only to 2^53)."""
    return str(value)


def _rsa_trace_json(trace: dict) -> dict:
    steps = trace["steps"]
    shown = steps[:MAX_RSA_STEPS_RETURNED]
    return {
        "base": _s(trace["base"]), "exponent": _s(trace["exponent"]), "modulus": _s(trace["modulus"]),
        "exponent_bits": trace["exponent_bits"], "initial": _s(trace["initial"]),
        "squarings": trace["squarings"], "multiplications": trace["multiplications"],
        "steps": [{"index": st["index"], "bit": st["bit"], "before": _s(st["before"]),
                   "after_square": _s(st["after_square"]), "multiplied": st["multiplied"],
                   "after_multiply": _s(st["after_multiply"])} for st in shown],
        "steps_total": len(steps), "steps_returned": len(shown), "steps_truncated": len(shown) < len(steps),
        "result": _s(trace["result"]),
    }


def _primality_json(n: int) -> dict:
    rep = encryption_rsa.primality_report(n)
    return {"is_prime": rep["is_prime"], "method": rep["method"], "bases": rep["bases"],
            "random_bases": rep["random_bases"], "error_bound": rep["error_bound"], "witness": rep["witness"]}


@app.post("/api/encrypt/rsa-keyexchange", response_model=RSAKeyExchangeResponse, tags=["stage 5: RSA"])
def rsa_key_exchange(req: RSAKeyExchangeRequest) -> RSAKeyExchangeResponse:
    """INSECURE educational RSA: key generation, key transport (or a small integer) and decryption, with all arithmetic."""
    wrap = req.mode == "aes_key_wrap"
    seeded = req.seed is not None
    rng = random.Random(req.seed) if seeded else secrets.SystemRandom()
    e_in = int(req.e) if req.e is not None else None
    try:
        if req.p is not None:
            p_in, q_in = int(req.p), int(req.q)
            for name, value in (("p", p_in), ("q", q_in)):
                if value.bit_length() > MAX_RSA_PRIME_BITS:
                    raise encryption_rsa.RSAKeyError(f"{name} is too large (at most {MAX_RSA_PRIME_BITS} bits)", name)
            pair, key_source = encryption_rsa.keypair_from_primes(p_in, q_in, e_in), "user_primes"
        else:
            bits = req.modulus_bits or (DEFAULT_RSA_WRAP_BITS if wrap else DEFAULT_RSA_INTEGER_BITS)
            if wrap and bits < MIN_RSA_WRAP_BITS:
                raise encryption_rsa.RSAKeyError(
                    f"modulus_bits must be at least {MIN_RSA_WRAP_BITS} to carry a 128-bit AES key "
                    f"(n must exceed 2^128)", "bits")
            pair, key_source = encryption_rsa.generate_keypair(bits, e_in, rng), "generated"
    except encryption_rsa.RSAKeyError as exc:
        field = _RSA_FIELD_MAP.get(exc.field, exc.field)
        raise ValidationFailure(str(exc), [{"field": f"body.{field}", "message": str(exc),
                                            "code": "invalid_key_parameter"}] if field else []) from exc

    pub, priv, n = pair.public, pair.private, pair.public.n
    key_bytes = None
    key_supplied = wrap and req.aes_key_hex is not None
    if wrap:
        key_bytes = bytes.fromhex(req.aes_key_hex) if key_supplied else rng.randbytes(16)
        m = encryption_rsa.key_to_int(key_bytes)
        if m >= n:
            where = "p" if req.p is not None else "modulus_bits"
            raise ValidationFailure(
                f"the AES key ({m.bit_length()} bits as an integer) does not fit below the modulus "
                f"({n.bit_length()} bits): raw RSA cannot carry it; use a larger modulus",
                [{"field": f"body.{where}", "message": "modulus too small for a 128-bit key", "code": "modulus_too_small"}])
        ciphertext = encryption_rsa.rsa_wrap_key(key_bytes, pub)
    else:
        if req.message is not None:
            m = int(req.message)
            if m >= n:
                raise ValidationFailure(
                    f"message must be smaller than the modulus n ({n.bit_length()} bits)",
                    [{"field": "body.message", "message": "message >= n", "code": "message_too_large"}])
        else:
            m = rng.randrange(2, n - 1)
        ciphertext = encryption_rsa.rsa_encrypt(m, pub)

    etrace = encryption_rsa.rsa_encrypt_trace(m, pub)
    dtrace = decryption_rsa.rsa_decrypt_trace(ciphertext, priv)
    recovered = decryption_rsa.rsa_decrypt(ciphertext, priv)
    key_recovered_ok, recovered_key = None, None
    if wrap:
        try:
            recovered_key = decryption_rsa.rsa_unwrap_key(ciphertext, priv, 16)
            key_recovered_ok = recovered_key == key_bytes
        except ValueError:
            key_recovered_ok = False

    p, q, phi, e, d = pair.p, pair.q, pair.phi, pub.e, priv.d
    checks = {
        "p_is_prime": encryption_rsa.is_prime(p), "q_is_prime": encryption_rsa.is_prime(q),
        "p_differs_from_q": p != q, "n_equals_p_times_q": n == p * q,
        "phi_equals_p1_times_q1": phi == (p - 1) * (q - 1), "e_less_than_phi": e < phi,
        "gcd_e_phi": _s(encryption_rsa.extended_gcd(e, phi)[0]), "e_times_d_mod_phi": _s(e * d % phi),
    }
    checks["all_ok"] = all(v for k, v in checks.items() if isinstance(v, bool)) and checks["gcd_e_phi"] == "1" \
        and checks["e_times_d_mod_phi"] == "1"
    shown_steps = pair.d_steps[:MAX_RSA_STEPS_RETURNED]
    d_derivation = {
        "method": "extended Euclid on (e mod phi, phi): d = x mod phi where e*x + phi*y = 1",
        "equation": f"e * d = 1 (mod phi):  {e} * {d} mod {phi} = {e * d % phi}",
        "steps": [{"quotient": _s(st["quotient"]), "remainder": _s(st["remainder"]), "x": _s(st["x"]), "y": _s(st["y"])}
                  for st in shown_steps],
        "steps_total": len(pair.d_steps), "steps_returned": len(shown_steps),
        "steps_truncated": len(shown_steps) < len(pair.d_steps),
    }
    round_trip_ok = recovered == m and dtrace["result"] == recovered
    expose_message = (not wrap) or (not key_supplied)
    return RSAKeyExchangeResponse(
        mode=req.mode, insecure=True, security_note=encryption_rsa.INSECURE_NOTE,
        real_world_minimum_modulus_bits=2048, textbook_notes=RSA_TEXTBOOK_NOTES,
        roles={"receiver": "generates the key pair and publishes the public key (n, e)",
               "sender": "encrypts the message / AES key with the public key: c = m^e mod n",
               "receiver_decrypts": "recovers m = c^d mod n with the private exponent d"},
        key_source=key_source, seeded=seeded, modulus_bits=n.bit_length(),
        p=_s(p), q=_s(q), n=_s(n), phi=_s(phi), e=_s(e), d=_s(d), e_source=pair.e_source,
        key_checks=checks, primality={"p": _primality_json(p), "q": _primality_json(q)},
        d_derivation=d_derivation,
        message_kind="aes_key_128" if wrap else "integer",
        message=_s(m) if expose_message else None, message_bits=m.bit_length(),
        aes_key_hex=key_bytes.hex() if wrap and not key_supplied else None,
        recovered_key_hex=recovered_key.hex() if wrap and not key_supplied and recovered_key is not None else None,
        ciphertext=_s(ciphertext), recovered=_s(recovered) if expose_message else None,
        encryption=_rsa_trace_json(etrace) if req.include_traces else None,
        decryption=_rsa_trace_json(dtrace) if req.include_traces else None,
        verification={"round_trip_ok": round_trip_ok, "key_recovered_ok": key_recovered_ok,
                      "encryption_trace_matches_ciphertext": etrace["result"] == ciphertext,
                      "key_checks_all_ok": checks["all_ok"]},
        round_trip_ok=round_trip_ok, key_recovered_ok=key_recovered_ok,
    )


HEADER_CHECKSUM_NOTE = (
    "The header checksum is the 16-bit Internet checksum (RFC 1071) over the 14 header bytes. It detects every "
    "single-bit header error but NOT reordered 16-bit words, cancelling errors, or any change to the payload; "
    "the CRC-32 stage that follows covers those.")


def _packet_record(position: int, packet, status: str) -> dict:
    preview = packet.payload[:MAX_PAYLOAD_PREVIEW_BYTES]
    return {
        "position": position, "sequence": packet.sequence, "total": packet.total,
        "payload_length": packet.payload_length, "serialized_size": packet.serialized_size,
        "header_hex": packet.header_bytes().hex(), "checksum": packet.checksum,
        "checksum_hex": f"0x{packet.checksum:04x}", "checksum_valid": packet.header_ok(), "status": status,
        "payload_hex": preview.hex(), "payload_truncated": len(preview) < packet.payload_length,
    }


def _packet_table(packets) -> list[dict]:
    """Records in arrival order; a later copy of an already-seen valid sequence is marked 'duplicate'."""
    seen: set[int] = set()
    table = []
    for pos, p in enumerate(packets[:MAX_PACKETS_RETURNED]):
        if not p.header_ok():
            status = "corrupted_header"
        elif p.sequence in seen:
            status = "duplicate"
        else:
            status = "ok"
            seen.add(p.sequence)
        table.append(_packet_record(pos, p, status))
    return table


@app.post("/api/packetize", response_model=PacketizeResponse, tags=["stage 6: packetization"])
def packetize(req: PacketizeRequest) -> PacketizeResponse:
    """Packetize data, optionally run a transmission experiment, reassemble, and report what was detected."""
    data = req.text.encode("utf-8") if req.text is not None else bytes.fromhex(req.data_hex)
    try:
        packets = packetization.create_packets(data, req.payload_size)
    except ValueError as exc:
        raise ValidationFailure(str(exc)) from exc
    count = len(packets)
    for name in ("drop_sequences", "duplicate_sequences", "corrupt_header_sequences"):
        bad = sorted({s for s in getattr(req, name) if s >= count})
        if bad:
            raise ValidationFailure(
                f"{name}: sequence {bad[0]} does not exist (the message has {count} packet(s), numbered 0..{count - 1})",
                [{"field": f"body.{name}", "message": "sequence out of range", "code": "sequence_out_of_range"}])

    try:                                                                  # the untouched packets must round-trip
        lossless_ok = packetization.reassemble_packets(packets) == data
    except packetization.PacketError:                                     # a server-side inconsistency, not a user error
        lossless_ok = False

    # --- controlled experiment (deterministic order; see PacketizeRequest)
    items = list(enumerate(packets))                                      # (original sequence, packet)
    corrupted = sorted(set(req.corrupt_header_sequences))
    for s in corrupted:
        items[s] = (s, packetization.corrupt_header(packets[s]))
    duplicated = sorted(set(req.duplicate_sequences))
    dup_set = set(duplicated)
    items = [it for orig, p in items for it in ([(orig, p), (orig, p)] if orig in dup_set else [(orig, p)])]
    explicit_drops = sorted(set(req.drop_sequences))
    drop_set = set(explicit_drops)
    items = [it for it in items if it[0] not in drop_set]
    survivors, lost_positions = packetization.drop_packets([p for _, p in items], req.loss_probability, req.seed)
    random_lost = sorted({items[i][0] for i in lost_positions})
    lost_pos = set(lost_positions)
    items = [it for i, it in enumerate(items) if i not in lost_pos]
    if req.shuffle:
        random.Random(req.seed).shuffle(items)
    received = [p for _, p in items]

    allow = req.allow_identical_duplicates
    diagnosis = packetization.inspect_packets(received, allow)
    try:
        recovered = packetization.reassemble_packets(received, allow)
        reassembly = {"success": True, "error_kind": None, "message": None, "details": {},
                      "recovered_bytes": len(recovered), "recovered_matches_original": recovered == data}
    except packetization.PacketError as exc:
        reassembly = {"success": False, "error_kind": exc.kind, "message": str(exc), "details": exc.details,
                      "recovered_bytes": None, "recovered_matches_original": None}
    experiment = {
        "applied": bool(corrupted or duplicated or explicit_drops or req.loss_probability > 0 or req.shuffle),
        "corrupted_header_sequences": corrupted, "duplicated_sequences": duplicated,
        "dropped_sequences_explicit": explicit_drops, "dropped_sequences_random": random_lost,
        "loss_probability": req.loss_probability, "seed": req.seed, "shuffled": req.shuffle,
        "sent_count": count, "received_count": len(received),
    }
    table = _packet_table(packets)
    received_table = _packet_table(received)
    return PacketizeResponse(
        data_bytes=len(data), payload_size=req.payload_size, header_size=packetization.HEADER_SIZE,
        stats=packetization.packetization_stats(packets),
        packets=table, packets_returned=len(table), packets_truncated=len(table) < count,
        lossless_round_trip_ok=lossless_ok, experiment=experiment,
        received_packets=received_table, received_truncated=len(received_table) < len(received),
        diagnosis=diagnosis, reassembly=reassembly, header_checksum_note=HEADER_CHECKSUM_NOTE,
    )


CRC_PROPERTIES = [
    "A CRC DETECTS errors; it never corrects them. A mismatch says something changed, not where or how.",
    "CRC-32 detects every single-bit error and every error burst of 32 bits or fewer, because an error polynomial of "
    "degree below 32 can never be a nonzero multiple of the generator polynomial.",
    "Other error patterns escape detection with probability about 2^-32 (2.3e-10) for random damage; specific patterns "
    "that are multiples of the generator polynomial are invisible.",
    "A CRC is linear over GF(2), so it is NOT authentication: anyone can compute 4 bytes that force any chosen CRC "
    "(see 'forge_to_match_original'). Use a MAC such as HMAC for tamper protection.",
]
CRC_ESCAPE_PROBABILITY = 2.0 ** -32


def _net_flipped_bits(before: bytes, after: bytes) -> list[int]:
    return [8 * i + k for i, (a, b) in enumerate(zip(before, after)) if a != b for k in range(8) if ((a ^ b) >> k) & 1]


@app.post("/api/error-detect/crc", response_model=CRCResponse, tags=["stage 7: error detection"])
def crc_error_detection(req: CRCRequest) -> CRCResponse:
    """CRC-32 with its derivation, optional verification, deliberate corruption and CRC forging."""
    crc = error_detection_crc
    data = req.text.encode("utf-8") if req.text is not None else bytes.fromhex(req.data_hex)
    checksum = crc.crc32_compute(data)
    le = checksum.to_bytes(4, "little")
    cross = {"bitwise_matches_table": crc.crc32_bitwise(data) == checksum,
             "polynomial_division_matches": crc.crc32_polynomial_division(data) == checksum}
    cross["all_agree"] = all(cross.values())

    verification = None
    if req.expected_crc_hex is not None:
        expected = int(req.expected_crc_hex, 16)
        verification = {"expected_crc32": expected, "expected_crc32_hex": f"0x{expected:08x}",
                        "computed_crc32_hex": f"0x{checksum:08x}", "matches": crc.crc32_verify(data, expected)}

    corruption = forged = None
    if req.flip_bits or req.burst_length is not None:
        nbits = 8 * len(data)
        bad = sorted({b for b in req.flip_bits if b >= nbits})
        if bad:
            raise ValidationFailure(
                f"flip_bits: bit {bad[0]} is outside the data ({nbits} bits)",
                [{"field": "body.flip_bits", "message": "bit out of range", "code": "bit_out_of_range"}])
        burst = []
        if req.burst_length is not None:
            if req.burst_start_bit + req.burst_length > nbits:
                raise ValidationFailure(
                    f"the burst (bits {req.burst_start_bit}..{req.burst_start_bit + req.burst_length - 1}) "
                    f"extends beyond the data ({nbits} bits)",
                    [{"field": "body.burst_start_bit", "message": "burst out of range", "code": "burst_out_of_range"}])
            burst = list(range(req.burst_start_bit, req.burst_start_bit + req.burst_length))
        corrupted = crc.flip_bits(crc.flip_bits(data, req.flip_bits), burst)
        net = _net_flipped_bits(data, corrupted)
        span = crc.error_span(net)
        corrupted_crc = crc.crc32_compute(corrupted)
        detected = corrupted_crc != checksum
        guaranteed = None if span is None else span <= 32
        corruption = {
            "applied": True, "net_change": bool(net), "flipped_bit_count": len(net),
            "flipped_bits": net[:MAX_FLIPPED_BITS_LISTED], "flipped_bits_truncated": len(net) > MAX_FLIPPED_BITS_LISTED,
            "changed_byte_indices": sorted({p // 8 for p in net})[:64],
            "error_span_bits": span, "guaranteed_detection": guaranteed,
            "guarantee_consistent": (not guaranteed) or detected,
            "original_crc32": checksum, "original_crc32_hex": f"0x{checksum:08x}",
            "corrupted_crc32": corrupted_crc, "corrupted_crc32_hex": f"0x{corrupted_crc:08x}",
            "detected": detected, "verified_against_original": crc.crc32_verify(corrupted, checksum),
            "corrupted_data_hex_preview": corrupted[:128].hex(), "corrupted_data_truncated": len(corrupted) > 128,
            "escape_probability_for_random_errors": CRC_ESCAPE_PROBABILITY,
        }
        if req.forge_to_match_original:
            suffix = crc.crc32_forge_suffix(corrupted, checksum)
            forged_data = corrupted + suffix
            forged_crc = crc.crc32_compute(forged_data)
            forged = {
                "applied": True, "suffix_hex": suffix.hex(), "length_changed_by_bytes": 4,
                "forged_crc32": forged_crc, "forged_crc32_hex": f"0x{forged_crc:08x}",
                "matches_original": forged_crc == checksum, "detected_by_crc": forged_crc != checksum,
                "content_changed_before_forging": corrupted != data,
                "note": "The CRC was repaired by appending 4 computed bytes: a CRC cannot authenticate data.",
            }
    return CRCResponse(
        data_bytes=len(data), crc32=checksum, crc32_hex=f"0x{checksum:08x}", crc32_little_endian_hex=le.hex(),
        parameters={"width": 32, "polynomial_normal_hex": f"0x{crc.POLY_NORMAL:08x}",
                    "polynomial_reflected_hex": f"0x{crc.POLY_REFLECTED:08x}", "init_hex": f"0x{crc.INIT:08x}",
                    "xorout_hex": f"0x{crc.XOROUT:08x}", "reflect_input": True, "reflect_output": True,
                    "check_input": crc.CHECK_INPUT.decode("ascii"), "check_value_hex": f"0x{crc.CHECK_VALUE:08x}"},
        polynomial={"expression": crc.polynomial_expression(), "terms": crc.polynomial_terms(),
                    "degree": 32, "full_hex": f"0x{crc.POLY_FULL:x}"},
        cross_checks=cross,
        standard_check_value_ok=crc.crc32_compute(crc.CHECK_INPUT) == crc.CHECK_VALUE,
        residue_check=crc.crc32_compute(data + le) == crc.RESIDUE,
        verification=verification,
        trace=crc.crc32_trace(data, req.trace_bytes, req.bit_trace_bytes),
        table_hex=[f"0x{v:08x}" for v in crc.crc32_table()] if req.include_table else None,
        corruption=corruption, forged=forged, properties=CRC_PROPERTIES,
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

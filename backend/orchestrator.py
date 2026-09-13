"""
orchestrator.py — Runs the full 11-stage pipeline end-to-end.

Sender side:  Huffman -> byte-align -> LZ77 -> AES/RSA -> packetize ->
              CRC-32 -> (route lookup) -> ECC-encode (redundancy added)
              -> noisy channel
Receiver side: ECC-decode (fix bit errors) -> CRC-32 verify -> AES/RSA
              decrypt -> LZ77 decompress -> Huffman decode -> compare

Every stage appends a trace entry {stage_name, input_snapshot,
output_snapshot, math_detail} so the frontend can render a step-through
UI without re-calling each endpoint individually. The error-correction
ENCODING step (adding Hamming/Reed-Solomon redundancy) conceptually
belongs right before the channel -- it is included inside the "Error
Correction" stage's math_detail/output rather than as its own numbered
stage, since the spec's 11-stage list places "Error Correction" as a
single stage after "Channel Transmission".
"""

from __future__ import annotations
import secrets

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

RS_K = 16
RS_N_PARITY = 4
RSA_INITIAL_BITS = 72


def run_pipeline(
    message: str,
    noise_probability: float = 0.02,
    error_correction_scheme: str = "hamming",
    topology_source: str = "A",
    topology_target: str = "F",
    channel_type: str = "bsc",
    seed: int | None = None,
) -> dict:
    trace: list[dict] = []

    def add_stage(name: str, inp: dict, outp: dict, detail: str) -> None:
        trace.append({"stage_name": name, "input_snapshot": inp, "output_snapshot": outp, "math_detail": detail})

    # ============================== SENDER SIDE ==============================

    # ---- Stage 1: Source Encoding (Huffman) ----
    huffman_report = se.huffman_encode_report(message)
    huffman_bits = huffman_report["bitstring"]
    code_table = huffman_report["code_table"]
    add_stage(
        "1. Source Encoding (Huffman)",
        {"text": message, "length_chars": len(message)},
        {
            "bitstring_preview": huffman_bits[:120] + ("..." if len(huffman_bits) > 120 else ""),
            "entropy_bits_per_symbol": huffman_report["entropy"],
            "avg_code_length_bits_per_symbol": huffman_report["avg_code_length"],
            "original_bits_fixed_width": huffman_report["original_bits"],
            "encoded_bits": huffman_report["encoded_bits"],
            "num_distinct_symbols": len(code_table),
            "code_table": code_table,
        },
        "Shannon entropy H(X) = -sum p(x)log2 p(x) is the theoretical minimum average bits/symbol; "
        "Huffman's greedy min-heap merge achieves H(X) <= L < H(X)+1.",
    )

    # ---- Stage 2: Binary Representation ----
    pad_bits = (-len(huffman_bits)) % 8
    padded_huffman_bits = huffman_bits + "0" * pad_bits
    huffman_bytes = br.bits_to_bytes(padded_huffman_bits) if padded_huffman_bits else b""
    byte_grid = br.visualize_bits(padded_huffman_bits) if padded_huffman_bits else []
    add_stage(
        "2. Binary Representation",
        {"huffman_bitstring_length": len(huffman_bits)},
        {
            "pad_bits_added": pad_bits,
            "byte_aligned_length_bits": len(padded_huffman_bits),
            "num_bytes": len(huffman_bytes),
            "byte_grid_preview": byte_grid[:16],
        },
        "Regroup the Huffman bitstream into 8-bit bytes (base-2 -> base-256 positional regrouping) so later "
        "byte-oriented stages (LZ77, AES, CRC-32) can operate on it.",
    )

    # ---- Stage 3: Compression (LZ77) ----
    tokens = lz.lz77_compress(huffman_bytes)
    compressed_bytes = lz.serialize_tokens(tokens)
    ratio = (len(huffman_bytes) / len(compressed_bytes)) if len(compressed_bytes) > 0 else None
    add_stage(
        "3. Compression (LZ77)",
        {"input_size_bytes": len(huffman_bytes)},
        {
            "num_tokens": len(tokens),
            "serialized_size_bytes": len(compressed_bytes),
            "compression_ratio": round(ratio, 4) if ratio else None,
        },
        "Sliding-window LZ77 replaces repeated substrings with (offset,length) back-references into a bounded "
        "window, trading fixed per-token overhead against run length.",
    )

    # ---- Stage 4: Encryption (AES-128-CBC + RSA key exchange) ----
    aes_key = secrets.token_bytes(16)
    iv = secrets.token_bytes(16)
    ciphertext = aes.aes_cbc_encrypt(compressed_bytes, aes_key, iv)

    rsa_bits = RSA_INITIAL_BITS
    rsa_keys = None
    wrapped_key_int = None
    for _ in range(6):
        candidate = rsa.generate_keypair(bits=rsa_bits)
        try:
            wrapped_key_int = rsa.wrap_aes_key(aes_key, candidate["public_key"])
            rsa_keys = candidate
            break
        except ValueError:
            rsa_bits += 8
    if rsa_keys is None:
        raise RuntimeError("could not generate an RSA modulus large enough to wrap the AES-128 session key")

    add_stage(
        "4. Encryption (AES-128-CBC + RSA key exchange)",
        {"plaintext_size_bytes": len(compressed_bytes)},
        {
            "ciphertext_hex_preview": ciphertext.hex()[:64] + ("..." if len(ciphertext) > 32 else ""),
            "ciphertext_size_bytes": len(ciphertext),
            "aes_key_hex": aes_key.hex(),
            "iv_hex": iv.hex(),
            "rsa_p": rsa_keys["p"],
            "rsa_q": rsa_keys["q"],
            "rsa_n": rsa_keys["n"],
            "rsa_e": rsa_keys["e"],
            "rsa_d": rsa_keys["d"],
            "wrapped_aes_key_int": wrapped_key_int,
        },
        "AES: 10-round substitution-permutation network over GF(2^8), CBC-chained. RSA: c=m^e mod n wraps the "
        "AES session key so only the holder of d=e^-1 mod phi(n) can recover it (Euler's theorem).",
    )

    # ---- Stage 5: Packetization ----
    packets = pkt.create_packets(ciphertext, payload_size=64)
    reassembled = pkt.reassemble_packets(packets)
    assert reassembled == ciphertext, "internal error: packet reassembly is not lossy in this simulator"
    add_stage(
        "5. Packetization",
        {"data_size_bytes": len(ciphertext)},
        {
            "num_packets": len(packets),
            "payload_size": 64,
            "packet_preview": [
                {
                    "seq_num": p["seq_num"],
                    "total_packets": p["total_packets"],
                    "payload_size": len(p["payload"]),
                    "header_checksum": p["header_checksum"],
                }
                for p in packets[:5]
            ],
        },
        "Ciphertext is split into fixed-size payloads, each tagged with a sequence number so the receiver "
        "can reorder and reassemble them (sort by seq_num, concatenate).",
    )

    # ---- Stage 6: Error Detection (CRC-32) ----
    checksum = crc.crc32_compute(ciphertext)
    add_stage(
        "6. Error Detection (CRC-32)",
        {"data_size_bytes": len(ciphertext)},
        {"checksum_hex": format(checksum, "08x")},
        "CRC-32 = remainder of message(x)*x^32 divided by generator G(x)=0x04C11DB7 over GF(2); the receiver "
        "recomputes this after error correction to independently verify integrity.",
    )

    # ---- Stage 7: Routing (Dijkstra) ----
    topo = rd.default_topology()
    src_node = topology_source if topology_source in topo.adjacency else "A"
    tgt_node = topology_target if topology_target in topo.adjacency else topo.nodes()[-1]
    path, total_cost, relax_steps = rd.dijkstra(topo, src_node, tgt_node)
    add_stage(
        "7. Routing (Dijkstra)",
        {"source": src_node, "target": tgt_node, "topology": topo.to_dict()},
        {"path": path, "total_cost": total_cost, "num_relaxation_steps": len(relax_steps)},
        "Dijkstra greedily fixes the closest unvisited vertex and relaxes dist[v] = min(dist[v], dist[u]+w(u,v)); "
        "safe for non-negative weights since no unvisited vertex could then offer a cheaper path.",
    )

    # ---- Error-correction ENCODING (redundancy added before the channel) ----
    scheme = error_correction_scheme
    if scheme == "hamming":
        ciphertext_bits = br.bytes_to_bits(ciphertext)
        ecc_tx_bits = hm.hamming_encode_message(ciphertext_bits)
    else:
        rs_encoded_bytes = rs.rs_encode_message(ciphertext, k=RS_K, n_parity=RS_N_PARITY)
        ecc_tx_bits = br.bytes_to_bits(rs_encoded_bytes)

    # ---- Stage 8: Channel Transmission ----
    if channel_type == "awgn":
        sigma = max(noise_probability, 0.001) * 10.0
        channel_result = ch.awgn_transmit(ecc_tx_bits, sigma=sigma, seed=seed)
    else:
        channel_result = ch.bsc_transmit(ecc_tx_bits, noise_probability, seed=seed)
    received_bits = channel_result["received_bits"]
    preview_len = 160
    add_stage(
        "8. Channel Transmission",
        {
            "transmitted_bits_length": len(ecc_tx_bits),
            "channel_type": channel_type,
            "flip_probability_or_sigma": noise_probability,
            "transmitted_bits_preview": ecc_tx_bits[:preview_len],
        },
        {
            "num_flips": channel_result["num_flips"],
            "ber_observed": channel_result["ber_observed"],
            "ber_theoretical": channel_result["ber_theoretical"],
            "received_bits_preview": received_bits[:preview_len],
            "positions_flipped_preview": [p for p in channel_result["positions_flipped"] if p < preview_len],
        },
        "BSC: each bit flips independently with probability p (i.i.d. Bernoulli(p) trials). "
        "AWGN: BPSK symbols (+-1) plus N(0,sigma^2) noise, thresholded at 0; theoretical BER = Q(1/sigma).",
    )

    # ============================== RECEIVER SIDE ==============================

    recovered_ciphertext = None
    if scheme == "hamming":
        decoded_bits, block_error_flags = hm.hamming_decode_message(received_bits, original_length=len(ciphertext_bits))
        try:
            recovered_ciphertext = br.bits_to_bytes(decoded_bits)
        except ValueError:
            recovered_ciphertext = None
        ecc_output = {
            "scheme": "hamming",
            "num_blocks": len(block_error_flags),
            "blocks_with_corrected_errors": sum(1 for f in block_error_flags if f),
        }
        ecc_detail = (
            "Hamming(7,4): the 3-bit syndrome, computed via the parity-check matrix H, equals the 1-indexed "
            "position of any single flipped bit in each 7-bit block -- flip it to correct."
        )
    else:
        received_bytes = br.bits_to_bytes(received_bits)
        recovered_ciphertext, block_success_flags = rs.rs_decode_message(
            received_bytes, k=RS_K, n_parity=RS_N_PARITY, original_length=len(ciphertext)
        )
        ecc_output = {
            "scheme": "reed-solomon",
            "num_blocks": len(block_success_flags),
            "blocks_corrected_successfully": sum(block_success_flags),
            "max_correctable_symbol_errors_per_block": rs.MAX_CORRECTABLE_ERRORS,
        }
        ecc_detail = (
            f"Reed-Solomon over GF(2^8): Peterson-Gorenstein-Zierler solves a small linear system for the "
            f"error-locator polynomial, correcting up to {rs.MAX_CORRECTABLE_ERRORS} symbol errors per block "
            f"regardless of how many bits within each symbol are wrong (burst-error resilience)."
        )

    add_stage(
        "9. Error Correction",
        {"received_bits_length": len(received_bits), "scheme": scheme},
        ecc_output,
        ecc_detail,
    )

    crc_ok = crc.crc32_verify(recovered_ciphertext, checksum) if recovered_ciphertext is not None else False

    success = False
    recovered_text = None
    diff_positions: list[int] = []
    failure_reason = None

    try:
        if recovered_ciphertext is None:
            raise ValueError("error correction could not reconstruct a byte-aligned ciphertext")

        # ---- Stage 10: Decryption ----
        aes_key_recovered = rsa.unwrap_aes_key(wrapped_key_int, rsa_keys["private_key"], key_length_bytes=16)
        decrypted_compressed = aes_dec.aes_cbc_decrypt(recovered_ciphertext, aes_key_recovered, iv)
        add_stage(
            "10. Decryption (AES + RSA)",
            {"ciphertext_size_bytes": len(recovered_ciphertext)},
            {
                "recovered_plaintext_size_bytes": len(decrypted_compressed),
                "session_key_recovered_correctly": aes_key_recovered == aes_key,
            },
            "AES inverse cipher runs InvShiftRows/InvSubBytes/AddRoundKey/InvMixColumns in reverse round order; "
            "RSA m=c^d mod n recovers the AES session key via Euler's theorem.",
        )

        # ---- Stage 11: Decoding ----
        recovered_tokens = lz.deserialize_tokens(decrypted_compressed)
        recovered_huffman_bytes = lz.lz77_decompress(recovered_tokens)
        recovered_padded_bits = br.bytes_to_bits(recovered_huffman_bytes)
        recovered_huffman_bits = recovered_padded_bits[: len(recovered_padded_bits) - pad_bits] if pad_bits else recovered_padded_bits
        recovered_text = se.huffman_decode(recovered_huffman_bits, code_table)

        success = recovered_text == message
        if not success:
            max_len = max(len(recovered_text), len(message))
            diff_positions = [
                i for i in range(max_len)
                if (message[i] if i < len(message) else None) != (recovered_text[i] if i < len(recovered_text) else None)
            ]

        add_stage(
            "11. Decoding",
            {"compressed_size_bytes": len(decrypted_compressed)},
            {
                "recovered_text_preview": recovered_text[:200],
                "matches_original_byte_for_byte": success,
            },
            "Reverse LZ77 -> regroup bits -> Huffman-decode via the code table -> compare byte-for-byte "
            "against the original message: the pipeline's definitive success/failure verdict.",
        )
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    if not success and failure_reason is None:
        failure_reason = "CRC-32 mismatch after error correction: errors exceeded the scheme's correction capacity."

    return {
        "success": success,
        "original_message": message,
        "recovered_message": recovered_text,
        "diff_positions": diff_positions,
        "failure_reason": None if success else failure_reason,
        "crc_ok_post_correction": crc_ok,
        "trace": trace,
    }

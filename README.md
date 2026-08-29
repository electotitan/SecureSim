# Secure Digital Communication Simulator

Phase 1 built the architecture. **Phase 2 fills it with real,
from-first-principles math** — every stage below actually does what its
name says, not a passthrough.

## What's here

```
secure-comm-sim/
├── core/
│   ├── pipeline.py        # Architecture: Stage, Trace, Pipeline, build_default_pipeline()
│   ├── encoding.py        # Binary encoding/decoding (UTF-8 <-> bits)
│   ├── huffman.py         # Huffman compression/decompression (manual tree, entropy calc)
│   ├── rsa.py             # RSA keygen/encrypt/decrypt (Miller-Rabin, ext. Euclid, modpow)
│   ├── packetization.py   # Packet split/reassemble
│   ├── crc.py             # CRC-8 error detection (manual GF(2) polynomial division)
│   ├── graph_routing.py   # Dijkstra shortest-path routing (manual, not a library call)
│   ├── channel.py         # Binary symmetric channel (per-bit noise, probability p)
│   └── hamming.py         # Hamming(7,4) encode + syndrome-based correction
├── viz/
│   └── index.html         # Bare debug harness (NOT the real Phase 4 UI)
├── tests/
│   ├── test_pipeline.py         # Phase 1 architecture tests (placeholder pipeline)
│   └── test_phase2_modules.py   # Phase 2 math tests + full pipeline integration
├── app.py                 # FastAPI server: POST /run -> full trace JSON
└── requirements.txt
```

## Pipeline order (and one deliberate change from the abstract)

```
text -> Binary Encoding -> Huffman Compression -> RSA Encryption
     -> Packetization -> Hamming Encode -> CRC Attach -> Graph Routing
     -> Channel (noise) -> CRC Check -> Hamming Correct -> Reassembly
     -> RSA Decryption -> Huffman Decompression -> Binary Decoding -> text
```

The abstract lists "Hamming Error Correction" as one stage near the end.
In a real pipeline this can't work as written — error correction only
means something if redundancy is **added before** the noisy channel and
**used after** it. So Hamming (and CRC) are each split into an
encode/attach half and a correct/check half that sandwich routing and the
channel. Every stage named in the abstract is still present; this is a
correctness fix to *where* they sit, not a change in scope.

## Proof it actually works

`tests/test_phase2_modules.py` includes:
- Every math primitive tested against known values (primality, Bézout
  identity, modular inverse, Shannon entropy of a uniform distribution).
- **Every one of the 7 possible single-bit-flip positions** in a
  Hamming(7,4) codeword is verified to self-correct exactly.
- A documented test of the code's real limitation: 2-bit errors in one
  block are *not* reliably corrected (this is true of Hamming(7,4), not a
  bug).
- Full pipeline round-trip with a clean channel (must recover exactly).
- Full pipeline under light noise (must recover *most* of the time).

Manually sweeping the noise probability (see below) shows exactly the
degradation curve you'd expect and want for your report's BER graph:

| p (flip probability) | success rate (20 trials) |
|---|---|
| 0.000 | 20/20 |
| 0.005 | 19/20 |
| 0.010 | 18/20 |
| 0.020 | 12/20 |
| 0.050 | 1/20 |
| 0.100 | 0/20 |

This is real, generated data — rerun it any time:
```bash
python3 -c "
from core.pipeline import build_default_pipeline
from core.rsa import generate_keypair
keypair = generate_keypair(16)
for p in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]:
    ok = sum(build_default_pipeline(channel_flip_probability=p, keypair=keypair)
             .run('Hello, secure world!')['recovered_correctly'] for _ in range(20))
    print(f'p={p:<6} success: {ok}/20')
"
```

## Running it

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/` for the debug harness, or:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, secure world!", "flip_probability": 0.02}'
```

Run tests:

```bash
pytest tests/ -v    # 27 tests, all passing
```

## Next step (Phase 3: integration & validation / Phase 4: real visualization)

The math is done and tested. What's left:
- **Phase 3**: broader edge-case testing (empty messages, unicode,
  very long messages, larger RSA keys), and a proper BER-vs-p chart
  generated from many trials for the report.
- **Phase 4**: replace `viz/index.html`'s raw JSON dump with real
  animated visuals per stage — the Huffman tree growing, RSA's modular
  exponentiation, the CRC long division, packets moving along the
  Dijkstra path, bits flipping red in the channel, and the Hamming
  syndrome catching (or missing) them. Every stage's `trace.steps`
  already contains exactly the data needed to drive each animation.

# SecureSim — Interactive Mathematical Simulator for Secure Digital Communication

A full-stack simulator that sends a text message through a complete 11-stage digital
communication pipeline — Huffman coding, LZ77 compression, AES-128 + RSA encryption,
packetization, CRC-32, Dijkstra routing, a noisy channel, Hamming(7,4) / Reed-Solomon error
correction, decryption, and decoding — with every mathematical transformation implemented from
first principles in Python and visualized stage by stage in the browser.

See **[REPORT.md](REPORT.md)** for the full mapping of each stage to its mathematical discipline,
algorithm, and governing formula, including the documented Reed-Solomon scope limitation.

## Requirements

- Python 3.11+
- A modern browser (the frontend is plain HTML/CSS/JS + D3 via CDN, no build step)

## Setup

```bash
cd project
pip install -r backend/requirements.txt
```

## Run

```bash
uvicorn backend.main:app --reload --port 8000
```

Then open **http://localhost:8000** in a browser.

## Using the UI

1. **Type a message** in the message field (or keep the default).
2. Choose an **error correction scheme** — Hamming(7,4) corrects a single bit error per 7-bit
   block; Reed-Solomon corrects up to 2 whole-*symbol* errors per 20-byte block regardless of how
   many bits inside each symbol are wrong, so it tolerates burst noise better.
3. Choose a **channel model** — binary symmetric (each bit flips independently) or AWGN
   (Gaussian-noise analog channel).
4. Drag the **noise level** slider (0.0–0.3). At 0 the message always arrives intact; push it
   higher to watch error correction get overwhelmed.
5. Pick a **route** (source → target) on the 6-node default network topology.
6. Press **Run full pipeline**. A result banner appears immediately (green = recovered
   successfully, red = transmission failed with a diff view), and the **signal path** row above it
   turns into 11 clickable stage markers.
7. Use **Step Through** (Prev / Next, or click any stage marker) to inspect what happened at every
   individual stage — the Huffman bitstring and entropy, the AES ciphertext and RSA key-exchange
   values, the CRC-32 checksum, the Dijkstra shortest path drawn on the network graph, which bits
   the channel flipped (highlighted red), and exactly how many symbol/bit errors were corrected.
8. The **Explore** section at the bottom lets you run four stages in isolation, outside of a full
   transmission: build a Huffman tree for arbitrary text, edit the network topology and watch the
   shortest path update, fire bits through the noisy channel directly, or encrypt one AES block and
   see its round keys.

## Project layout

```
backend/
  main.py            FastAPI app: one endpoint per stage + the orchestrator
  models.py           Pydantic request/response schemas
  orchestrator.py      Runs all 11 stages end-to-end, returns a full step-through trace
  pipeline/            One module per stage — every algorithm from first principles
frontend/             Vanilla HTML/CSS/JS + D3 (CDN), served as static files by FastAPI
tests/                 pytest suite — 96 tests, including known-answer vectors:
                        - AES-128 vs. the official FIPS-197 test vector
                        - CRC-32 vs. the canonical "123456789" → 0xCBF43926 check value
                        - end-to-end: 0% noise always succeeds, high noise demonstrates failure
REPORT.md              Stage → math discipline → algorithm → formula, for academic write-up
```

## Testing

```bash
cd project
python3 -m pytest tests/ -v
```

All 96 tests should pass. Notably:

- `tests/test_aes.py::test_fips197_known_answer_vector` — the from-scratch AES-128 implementation
  reproduces the official FIPS-197 known-answer ciphertext exactly.
- `tests/test_crc.py::test_known_test_vector_123456789` — CRC-32 matches the canonical published
  check value.
- `tests/test_pipeline_end_to_end.py` — the full 11-stage pipeline always succeeds at 0% channel
  noise and demonstrates graceful, clearly-reported failure under heavy noise, for both error
  correction schemes and both channel models.

## API

Every pipeline stage is exposed as its own REST endpoint (e.g. `POST /api/encode/huffman`,
`POST /api/encrypt/aes`, `POST /api/error-correct/reed-solomon`) in addition to the full-pipeline
orchestrator at `POST /api/pipeline/run`. Interactive API docs are available at
**http://localhost:8000/docs** once the server is running.

## What's implemented from scratch vs. what's a library

Implemented from first principles (see REPORT.md for the exact formula each one is built on):
Huffman coding, LZ77, GF(2⁸) arithmetic, the AES S-box (derived, not hardcoded), AES-128 (key
schedule, all four round transformations, CBC mode), RSA (Miller-Rabin primality testing, extended
Euclid, square-and-multiply modular exponentiation), CRC-32 (polynomial division, table derived
from the generator polynomial), Dijkstra's algorithm, the BSC/AWGN channel models, Hamming(7,4),
and Reed-Solomon with Peterson-Gorenstein-Zierler decoding.

Used only for infrastructure, never for the underlying algorithms: FastAPI/Uvicorn/Pydantic (the
web layer) and D3 (rendering the Huffman tree and network graph — the tree/graph data itself comes
entirely from this project's own code).

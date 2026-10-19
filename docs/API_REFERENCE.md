# API Reference

> Grows with each checkpoint. Interactive documentation: `/docs` (also `/api/docs`).

## Conventions

* All bodies are JSON (UTF-8). Unknown request fields are rejected (HTTP 422).
* Text-based endpoints accept at most **10,000 characters**; the global request-body cap is **1,000,000 bytes**
  (declared `Content-Length`; chunked uploads without a length are not capped at this layer, but the field limit
  still applies).
* Bitstrings are strings of the characters `0` and `1`.

### Error format (all endpoints)

```json
{"error": {"type": "validation_error", "message": "Request validation failed.",
           "details": [{"field": "body.text", "message": "Field required", "code": "missing"}]}}
```

| `type`                  | meaning                                             | typical HTTP |
|-------------------------|-----------------------------------------------------|--------------|
| `validation_error`      | malformed / out-of-range input, oversized body       | 422, 413     |
| `http_error`            | routing problems (unknown path, wrong method)        | 404, 405     |
| `integrity_failure`     | bad ciphertext length / padding (AES); later: checksum, packets | 422 |
| `transmission_failure`  | errors exceeded the correction capacity              | (later stages) |
| `unsupported_operation` | well-formed but unsupported request                  | 501          |
| `internal_error`        | unexpected server fault; no internals are exposed    | 500          |

Tracebacks and the submitted input values are never included in error responses.

## GET `/api/health`

`{"status": "ok", "app": "...", "version": "0.5.0"}`

## POST `/api/encode/huffman`

Request: `{"text": "ABBCCC"}`

Response (abridged; values from a real run):

```json
{
  "bitstring": "101111000",
  "code_table": {"C": "0", "A": "10", "B": "11"},
  "entropy": 1.4591479170272448,
  "avg_code_length": 1.5,
  "efficiency_percent": 97.27652780181631,
  "symbols": [{"symbol": "C", "code_point": "U+0043", "count": 3, "probability": 0.5,
               "code": "0", "code_length": 1}, "..."],
  "tree": {"symbol": null, "freq": 6, "left": {"...": "..."}, "right": {"...": "..."}},
  "original_length_chars": 6,
  "original_length_bits_utf8": 48,
  "encoded_bit_length": 9,
  "ratio_vs_utf8": 5.333333333333333,
  "table_overhead_included": false,
  "round_trip_ok": true,
  "single_symbol_convention": false
}
```

Notes:
* `efficiency_percent` and `ratio_vs_utf8` are `null` when undefined (empty input).
* `round_trip_ok` is computed by decoding the returned bitstring on the server and comparing with the input.
* `tree` is `null` for empty input; edges: `left` = bit 0, `right` = bit 1; leaves have a non-null `symbol`.
* Empty text returns an empty bitstring/table. A single distinct symbol gets the code `"0"`.

## POST `/api/encode/binary`

Request: `{"text": "Hé"}`

Response (abridged): `bitstring` (`010010001100001110101001`), `byte_count`, `character_count`, `bytes`
(per byte: `index`, `binary`, `bits`, `hex`, `decimal`, `role`, `character`, `char_index`, `char_start`),
`characters` (per character: `character`, `code_point`, `code_point_hex`, `utf8_hex`, `utf8_byte_count`),
`round_trip_ok`.

`role` is one of `ascii`, `lead-2`, `lead-3`, `lead-4`, `continuation`, `invalid`.


## POST `/api/compress/lz77`

Request: exactly **one** of `text` (UTF-8 encoded, at most 10,000 characters) or `data_hex` (raw bytes as an even-length
hex string, at most 40,000 bytes; digits `0-9a-fA-F` only, no prefix or whitespace). Optional integers
`window_size` (1..32768, default 4096) and `lookahead_size` (1..255, default 18); booleans, floats and numeric strings
are rejected.

```json
{"text": "abracadabra", "window_size": 4096, "lookahead_size": 18}
```

Response (values from a real run):

```json
{
  "window_size": 4096, "lookahead_size": 18,
  "original_bytes": 11, "token_count": 6, "literal_count": 3, "match_count": 3,
  "offset_bits": 13, "length_bits": 5, "payload_bits": 154, "header_bytes": 1,
  "compressed_bytes": 21, "compression_ratio": 0.5238095238095238,
  "space_saved_percent": -90.9090909090909, "expanded": true,
  "tokens": [ {"index": 5, "position": 7, "kind": "match-end", "offset": 7, "length": 4,
               "next_hex": "", "next_char": null, "covers": 4, "match_source_start": 0,
               "match_hex": "61627261", "overlapping": false, "window_start": 0,
               "lookahead_end": 11, "bits": 19}, "..." ],
  "tokens_returned": 6, "tokens_truncated": false,
  "round_trip_ok": true, "serialization_round_trip_ok": true,
  "size_note": "compressed_bytes is the size of the fixed-width serialisation ..."
}
```

Notes:
* Tokens are `(offset, length, next_byte)`; `kind` is `literal`, `match`, or `match-end` (end-of-input token with an empty
  `next_hex`). Offsets count backward from the token's `position`.
* All statistics cover **every** token; at most 2,000 per-token records are returned (`tokens_truncated` tells you).
* `compression_ratio` and `space_saved_percent` are `null` for empty input. `expanded: true` means the serialised form is
  larger than the input (common for short or non-repeating data).
* `round_trip_ok`: the server decompressed the tokens and compared them with the input bytes.
  `serialization_round_trip_ok`: serialize then deserialize reproduced the tokens and the size matches the statistics.
  Both are computed on every request, never assumed.
* Errors use the standard format; invalid fields are named in `details[].field` (for example `body.window_size`, or `body`
  for the "exactly one of text/data_hex" rule).


## POST `/api/encrypt/aes`

AES-128-CBC with PKCS#7 padding. Only AES-128 exists: `key_bits` other than 128 returns **HTTP 501**
(`unsupported_operation`); values other than 128/192/256 return 422.

Request (plaintext is exactly one of `text` or `data_hex`, at most 40,000 bytes):

| field | type | default | notes |
|-------|------|---------|-------|
| `text` / `data_hex` | string | | exactly one required; `data_hex` = even-length hex, no prefix or whitespace |
| `key_hex` | 32 hex digits | generated | the 16-byte key |
| `iv_hex` | 32 hex digits | generated | the 16-byte IV |
| `seed` | integer 0..2^32-1 | none | reproducible key/IV: **first** 16 bytes drawn from `random.Random(seed)` are the key, the next 16 the IV (always both drawn, whichever you supply). Insecure; for experiments |
| `key_bits` | integer | 128 | 128 only |
| `include_trace` | boolean | true | state after every transformation for one block |
| `trace_block` | integer >= 0 | 0 | which block to trace (the actual cipher input `P_i XOR C_(i-1)`) |
| `reveal_round_keys` | boolean | true | see secrets below |

Response (abridged): `algorithm`, `mode`, `padding`, `key_bits`, `key_hex` (only if the server generated it),
`key_generated`, `key_seeded`, `iv_hex`, `iv_generated`, `plaintext_bytes`, `padding_bytes`, `padded_bytes`, `block_count`,
`ciphertext_hex`, `blocks` (first 64 only, `blocks_truncated`; each has `plaintext_hex`, `cipher_input_hex`,
`ciphertext_hex`), `trace`, `trace_block`, `round_trip_ok`, `warning`.

* `round_trip_ok` is computed: the server decrypts the ciphertext and compares it with the input bytes.
* `trace`: `input_hex`, `output_hex`, `round_keys_hex`, `byte_order`, and 41 `steps`
  (`round`, `step`, `state` as a 4x4 matrix of rows, `state_hex` in block byte order). `SubBytes` steps carry per-byte
  `detail` (`input`, field `inverse`, `output`); `MixColumns` steps carry the GF(2^8) terms of every output byte.

**Secrets.** A key you supply is never returned in the response body *except* inside the trace: AddRoundKey needs the
round keys and round key 0 equals the key. Traces made with real keys must not be shared or stored. Send
`"reveal_round_keys": false` to redact every round-key field while keeping all state matrices. A generated key is
returned in `key_hex` (you need it to decrypt). The server does not log request bodies or key material.

## POST `/api/decrypt/aes`

Request: `ciphertext_hex` (even-length hex, at most 40,016 bytes), `key_hex`, `iv_hex` (all required), and optional
`include_trace`, `trace_block`, `reveal_round_keys` as above.

Response (abridged): `plaintext_hex`, `plaintext_bytes`, `plaintext_text` (UTF-8 decoding, or `null`),
`plaintext_is_valid_utf8`, `padding_bytes`, `block_count`, `blocks` (first 64: `ciphertext_hex`, `decrypted_hex` =
`D_K(C_i)`, `plaintext_hex` = `D_K(C_i) XOR C_(i-1)`), `trace` (inverse cipher, round numbers counting down 10..0),
`trace_block`, `warning`. The key is never returned.

Errors: an empty ciphertext or one whose length is not a multiple of 16, and malformed PKCS#7 padding, return HTTP 422 with
`"type": "integrity_failure"`. A wrong key or tampered ciphertext often (not always) shows up as a padding error;
CBC cannot distinguish "wrong key" from "corrupted data" reliably, and sometimes decrypts to garbage without any error.

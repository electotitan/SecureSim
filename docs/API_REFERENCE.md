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

`{"status": "ok", "app": "...", "version": "0.8.0"}`

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


## POST `/api/encrypt/rsa-keyexchange`

**Insecure educational RSA** (textbook RSA, tiny keys); every response has `insecure: true`. All large integers are exchanged
as **decimal strings** (JavaScript numbers are exact only up to 2^53). Generates (or inspects) a key pair, encrypts, decrypts, and
returns all the arithmetic.

| field | type | default | notes |
|-------|------|---------|-------|
| `mode` | `"aes_key_wrap"` or `"integer"` | `aes_key_wrap` | `aes_key_wrap` carries a 128-bit AES key; `integer` encrypts a small integer (not real key transport) |
| `modulus_bits` | even integer 12..512 | 160 (wrap) / 32 (integer) | wrap mode needs at least 130 so that n > 2^128; cannot be combined with `p`/`q` |
| `p`, `q` | decimal strings | generated | supply both or neither; at most 78 digits and 256 bits each; must be distinct odd primes |
| `e` | decimal string | 65537, else 257/17/5/3 | odd, 3 <= e < phi, coprime to phi |
| `message` | decimal string | random (integer mode) | integer mode only; must be < n; no signs, spaces or leading zeros |
| `aes_key_hex` | 32 hex digits | generated | wrap mode only |
| `seed` | integer 0..2^32-1 | none | reproducible: primes are drawn first (from `random.Random(seed)`, p then q), then the AES key or random message. Insecure |
| `include_traces` | boolean | true | omit the square-and-multiply traces |

Response (abridged): `mode`, `insecure`, `security_note`, `real_world_minimum_modulus_bits` (2048), `textbook_notes`, `roles`,
`key_source` (`generated` / `user_primes`), `seeded`, `modulus_bits`, `p`, `q`, `n`, `phi`, `e`, `d` (strings), `e_source`
(`default` / `fallback` / `user`), `key_checks` (computed: primality of p and q, `n = p*q`, `phi`, `gcd(e, phi)`, `e*d mod phi`,
`all_ok`), `primality` (per prime: method, bases, random bases, error bound, witness), `d_derivation` (the extended-Euclid
steps, bounded), `message`, `message_bits`, `aes_key_hex` / `recovered_key_hex` (only for a server-generated demo key),
`ciphertext`, `recovered`, `encryption` / `decryption` (square-and-multiply traces: `exponent_bits`, `initial`, `squarings`,
`multiplications`, `steps[]` with `before`, `after_square`, `multiplied`, `after_multiply`, `result`; at most 64 steps are returned and
`steps_total` / `steps_truncated` say how many exist), `verification` and `round_trip_ok` / `key_recovered_ok` (computed by
decrypting and comparing; never assumed).

**Secrets.** A key you supply is not echoed in `aes_key_hex`, `message` or `recovered`, but the traces contain the key as an integer
(`encryption.initial`, `decryption.result`); send `"include_traces": false` to omit them. A generated demo key is returned
because you need it. The server does not log request bodies or key material. The private exponent `d` (and `p`, `q`, `phi`) are
returned because showing them is the purpose of the demonstration; use disposable keys only.

**Errors** (HTTP 422, `validation_error`; `details[].field` names the parameter, and `details[].code` distinguishes
`invalid_key_parameter` from request-model codes such as `string_too_long` or `value_error`): `p`/`q` not prime or not odd, `p == q`,
`e` not coprime to phi or not smaller than phi, a message not smaller than `n`, a modulus too small to carry a 128-bit key (the key is
never truncated), odd or out-of-range `modulus_bits`, `p` without `q`, conflicting fields.


## POST `/api/packetize`

Splits data into packets (format and rules: `MATHEMATICAL_FOUNDATIONS.md` section 9), optionally perturbs the packet set, then runs the receiver:
diagnosis and reassembly.

| field | type | default | notes |
|-------|------|---------|-------|
| `text` / `data_hex` | string | | exactly one required; at most 40,000 bytes |
| `payload_size` | integer 1..4096 | 64 | maximum payload bytes per packet |
| `loss_probability` | number 0..1 | 0 | each packet is lost independently; integers 0 and 1 accepted, strings and booleans rejected |
| `seed` | integer 0..2^32-1 | none | makes random loss and shuffling reproducible |
| `drop_sequences` | list of integers | [] | original sequence numbers to lose (all copies); at most 1000 entries |
| `duplicate_sequences` | list of integers | [] | sequences to send twice |
| `corrupt_header_sequences` | list of integers | [] | flip bit 0 of the sequence field but keep the old checksum |
| `shuffle` | boolean | false | random arrival order |
| `allow_identical_duplicates` | boolean | false | tolerate repeated packets whose payload equals the first copy |

Sequence numbers in the three lists must exist (`0 .. packet_count-1`, HTTP 422 with `code: sequence_out_of_range` otherwise). Experiment order:
corrupt, duplicate, drop (explicit, then random), shuffle.

Response (abridged; values from a real run on the 37-byte message with `payload_size` 10):

```json
{
  "data_bytes": 37, "payload_size": 10, "header_size": 14,
  "stats": {"packet_count": 4, "payload_bytes": 37, "header_bytes_total": 56, "serialized_bytes_total": 93,
            "overhead_percent": 60.215, "efficiency_percent": 39.785, "largest_payload": 10, "last_payload": 7, "header_size": 14},
  "packets": [{"position": 0, "sequence": 0, "total": 4, "payload_length": 10, "serialized_size": 24,
               "header_hex": "01000000000000000004000afef1", "checksum": 65265, "checksum_hex": "0xfef1",
               "checksum_valid": true, "status": "ok", "payload_hex": "68656c6c6f207061636b", "payload_truncated": false}, "..."],
  "packets_returned": 4, "packets_truncated": false,
  "lossless_round_trip_ok": true,
  "experiment": {"applied": false, "sent_count": 4, "received_count": 4, "...": "..."},
  "received_packets": ["... same record shape, in ARRIVAL order ..."], "received_truncated": false,
  "diagnosis": {"status": "complete", "complete": true, "missing": [], "duplicate_sequences": [], "corrupted_positions": [], "...": "..."},
  "reassembly": {"success": true, "error_kind": null, "message": null, "details": {}, "recovered_bytes": 37, "recovered_matches_original": true},
  "header_checksum_note": "..."
}
```

Notes:
* `stats` describe the packets **sent**; `received_packets`/`diagnosis`/`reassembly` describe what **arrived**.
* Per-packet `status`: `ok`, `corrupted_header`, or `duplicate` (a later copy of an already-seen valid sequence).
* At most 256 packet records per table and 128 payload bytes per record are returned (`packets_truncated`, `received_truncated`, `payload_truncated`);
  statistics, diagnosis and the round trip always cover every packet.
* `lossless_round_trip_ok` is computed: the untouched packets are reassembled and compared with the input byte for byte. A failure of that
  self-check is reported as `false`, never as an HTTP error.
* `reassembly.error_kind` is one of `no_packets`, `header_checksum`, `inconsistent_total`, `invalid_sequence`, `duplicate`, `missing` (priority order in the
  foundations document). A failed reassembly is a normal 200 response, not an error: it is the experiment's result. `details` carries bounded lists plus exact
  counts (`missing`, `missing_count`, `missing_truncated`, ...).
* `diagnosis` ignores packets that fail their header checksum when computing totals, duplicates and missing sequences (they are listed under `corrupted_*`).
* `recovered_matches_original` compares the recovered bytes with the input byte for byte; success without a match would be reported as `false`.


## POST `/api/error-detect/crc`

Computes CRC-32 with its derivation, optionally verifies it, optionally corrupts the data on purpose, and can forge a CRC-matching message to show that a CRC is
not authentication. Algorithm and conventions: `MATHEMATICAL_FOUNDATIONS.md` section 10.

| field | type | default | notes |
|-------|------|---------|-------|
| `text` / `data_hex` | string | | exactly one required; at most 40,000 bytes |
| `expected_crc_hex` | 8 hex digits | none | if given, the data is verified against it (`verification`) |
| `flip_bits` | list of integers | [] | bit positions to flip; at most 1000 entries |
| `burst_start_bit`, `burst_length` | integers | none | flip every bit from the start for `burst_length` bits (1..4096); give both or neither |
| `forge_to_match_original` | boolean | false | after the corruption, append 4 computed bytes so the CRC equals the original's; needs `flip_bits` or a burst |
| `trace_bytes` | integer 0..256 | 16 | bytes of per-byte register steps returned |
| `bit_trace_bytes` | integer 0..8 | 2 | bytes of bit-level shift/XOR steps returned |
| `include_table` | boolean | true | return the derived 256-entry table |

**Bit positions** count in transmission order: bit `p` is bit `p mod 8` of byte `p div 8`, bit 0 being the least significant bit of the byte. All listed flips and the
burst are XOR-ed into the data, so a bit named twice (or by both a flip and the burst) is flipped back; the response reports the **net** change.

Response (abridged; values from a real run):

```json
{
  "data_bytes": 9, "crc32": 3421780262, "crc32_hex": "0xcbf43926", "crc32_little_endian_hex": "2639f4cb",
  "parameters": {"width": 32, "polynomial_normal_hex": "0x04c11db7", "polynomial_reflected_hex": "0xedb88320",
                 "init_hex": "0xffffffff", "xorout_hex": "0xffffffff", "reflect_input": true, "reflect_output": true,
                 "check_input": "123456789", "check_value_hex": "0xcbf43926"},
  "polynomial": {"expression": "x^32 + x^26 + ... + x + 1", "terms": [32, 26, 23, "..."], "degree": 32, "full_hex": "0x104c11db7"},
  "cross_checks": {"bitwise_matches_table": true, "polynomial_division_matches": true, "all_agree": true},
  "standard_check_value_ok": true, "residue_check": true,
  "verification": {"expected_crc32": 3421780262, "expected_crc32_hex": "0xcbf43926", "computed_crc32_hex": "0xcbf43926", "matches": true},
  "trace": {"initial_register": 4294967295, "byte_steps": [{"index": 0, "byte": 49, "register_before": 4294967295, "table_index": 206,
            "table_value": 2094854071, "register_after": 2083721288}], "bit_steps": ["..."], "steps_total": 9, "steps_returned": 9,
            "steps_truncated": false, "final_register": 872412870, "final_xor": 4294967295, "checksum": 3421780262},
  "table_hex": ["0x00000000", "0x77073096", "..."],
  "corruption": {"applied": true, "net_change": true, "flipped_bit_count": 1, "flipped_bits": [100], "error_span_bits": 1,
                 "guaranteed_detection": true, "guarantee_consistent": true, "original_crc32_hex": "0x414fa339",
                 "corrupted_crc32_hex": "0xaf312e29", "detected": true, "verified_against_original": false, "...": "..."},
  "forged": {"applied": true, "suffix_hex": "85ef20d7", "forged_crc32_hex": "0x414fa339", "matches_original": true,
             "detected_by_crc": false, "content_changed_before_forging": true, "length_changed_by_bytes": 4, "note": "..."},
  "properties": ["A CRC DETECTS errors; it never corrects them. ...", "..."]
}
```

Notes:
* `standard_check_value_ok`, `residue_check` and `cross_checks` are **recomputed on every request**: the check string `123456789` must give `0xCBF43926`, appending the
  little-endian CRC must give the residue `0x2144DF1C`, and the table, bit-by-bit and polynomial-division computations must agree.
* `corruption.guaranteed_detection` is true when the **net** error spans at most 32 bits (first to last flipped bit): CRC-32 detects every such burst. When it is false
  the CRC usually still detects the error but may not. `guarantee_consistent` is a self-check that a guaranteed detection really was detected. If you flip bits so they
  cancel (`net_change: false`), nothing changed and nothing is detected.
* `detected: false` is a legitimate outcome, not an error: for error patterns that are multiples of the generator polynomial the corruption escapes (see the worked 33-bit
  example in the foundations document).
* `forged` lengthens the data by 4 bytes. `content_changed_before_forging` is false when the requested flips cancelled out.
* Traces are bounded (`steps_total` / `steps_truncated` give the full size); the checksum always covers all the data. At most 1000 net flipped bit positions are listed
  (`flipped_bits_truncated`, `flipped_bit_count` is exact).
* Errors (HTTP 422, `validation_error`): a bit or burst outside the data (`body.flip_bits` / `body.burst_start_bit`, codes `bit_out_of_range` / `burst_out_of_range`), a burst
  with only one of its two fields, forging without a corruption, and malformed hex or numbers (the offending field is named).

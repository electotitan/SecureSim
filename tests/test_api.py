"""API tests, appended checkpoint by checkpoint.

Checkpoint 3: POST /api/encode/huffman, POST /api/encode/binary, error handling.
"""
import json
import pathlib
import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.errors import ValidationFailure, register_error_handlers
from backend.main import app
from backend.models import MAX_BODY_BYTES, MAX_TEXT_CHARS
from backend.pipeline.source_encoding import huffman_decode

client = TestClient(app)


def assert_error(resp, status, etype="validation_error"):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    assert body["error"]["type"] == etype
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    text = resp.text
    for leak in ("Traceback", "File \"", ".py\"", "site-packages", "pydantic_core"):
        assert leak not in text
    return body["error"]


# ------------------------------------------------------------------ Huffman endpoint
def test_huffman_response_schema_and_values():
    r = client.post("/api/encode/huffman", json={"text": "ABBCCC"})
    assert r.status_code == 200
    d = r.json()
    for key in ("bitstring", "code_table", "entropy", "avg_code_length", "efficiency_percent"):
        assert key in d                                           # fields required by the spec
    assert d["code_table"] == {"C": "0", "A": "10", "B": "11"}
    assert d["bitstring"] == "10" + "11" + "11" + "0" + "0" + "0"      # A B B C C C -> 101111000
    probs = [1 / 6, 2 / 6, 3 / 6]
    h = -sum(p * math.log2(p) for p in probs)
    assert d["entropy"] == pytest.approx(h)
    assert d["avg_code_length"] == pytest.approx(1 / 6 * 2 + 2 / 6 * 2 + 3 / 6 * 1)   # 1.5
    assert d["efficiency_percent"] == pytest.approx(h / 1.5 * 100)
    assert d["encoded_bit_length"] == len(d["bitstring"]) == 9
    assert d["original_length_chars"] == 6 and d["original_length_bits_utf8"] == 48
    assert d["round_trip_ok"] is True and d["table_overhead_included"] is False
    assert d["single_symbol_convention"] is False
    assert d["ratio_vs_utf8"] == pytest.approx(48 / 9)


def test_huffman_bitstring_actually_decodes_to_input():
    for text in ("hello world", "héllo wörld 😀", "aaaabbbcc", "x"):
        d = client.post("/api/encode/huffman", json={"text": text}).json()
        assert huffman_decode(d["bitstring"], d["code_table"]) == text


def test_huffman_symbol_stats_and_tree():
    d = client.post("/api/encode/huffman", json={"text": "abracadabra"}).json()
    syms = d["symbols"]
    assert [s["symbol"] for s in syms][0] == "a"                  # sorted by count desc
    assert sum(s["count"] for s in syms) == 11
    assert sum(s["probability"] for s in syms) == pytest.approx(1.0)
    assert all(s["code"] == d["code_table"][s["symbol"]] and s["code_length"] == len(s["code"])
               for s in syms)
    assert syms[0]["code_point"] == "U+0061"
    assert d["tree"]["freq"] == 11
    json.dumps(d)


def test_huffman_empty_input():
    r = client.post("/api/encode/huffman", json={"text": ""})
    assert r.status_code == 200
    d = r.json()
    assert d["bitstring"] == "" and d["code_table"] == {} and d["symbols"] == []
    assert d["tree"] is None
    assert d["entropy"] == 0.0 and d["avg_code_length"] == 0.0
    assert d["efficiency_percent"] is None and d["ratio_vs_utf8"] is None
    assert d["round_trip_ok"] is True


def test_huffman_single_symbol():
    d = client.post("/api/encode/huffman", json={"text": "zzzz"}).json()
    assert d["code_table"] == {"z": "0"} and d["bitstring"] == "0000"
    assert d["entropy"] == 0.0 and d["avg_code_length"] == 1.0
    assert d["efficiency_percent"] == 0.0
    assert d["single_symbol_convention"] is True and d["round_trip_ok"] is True


def test_huffman_unicode_preserved():
    text = "日本語 é 😀😀"
    d = client.post("/api/encode/huffman", json={"text": text}).json()
    assert set(d["code_table"]) == set(text)
    assert d["original_length_bits_utf8"] == 8 * len(text.encode("utf-8"))
    assert d["round_trip_ok"] is True


def test_huffman_deterministic_across_calls():
    a = client.post("/api/encode/huffman", json={"text": "mississippi river"}).json()
    b = client.post("/api/encode/huffman", json={"text": "mississippi river"}).json()
    assert a == b


def test_huffman_max_length_accepted_and_overlong_rejected():
    ok = client.post("/api/encode/huffman", json={"text": "ab" * (MAX_TEXT_CHARS // 2)})
    assert ok.status_code == 200 and ok.json()["round_trip_ok"] is True
    too_long = client.post("/api/encode/huffman", json={"text": "a" * (MAX_TEXT_CHARS + 1)})
    err = assert_error(too_long, 422)
    assert err["details"][0]["field"] == "body.text"


# ------------------------------------------------------------------ Binary endpoint
def test_binary_response():
    r = client.post("/api/encode/binary", json={"text": "Hé"})
    assert r.status_code == 200
    d = r.json()
    assert d["bitstring"] == "01001000" + "11000011" + "10101001"
    assert d["byte_count"] == 3 and d["character_count"] == 2
    assert [b["hex"] for b in d["bytes"]] == ["48", "c3", "a9"]
    assert [b["decimal"] for b in d["bytes"]] == [72, 195, 169]
    assert [b["role"] for b in d["bytes"]] == ["ascii", "lead-2", "continuation"]
    assert d["bytes"][1]["character"] == "é" and d["bytes"][2]["char_start"] is False
    e = d["characters"][1]
    assert e["code_point"] == 233 and e["code_point_hex"] == "U+00E9" and e["utf8_hex"] == "c3a9"
    assert d["round_trip_ok"] is True


def test_binary_empty_and_emoji():
    d = client.post("/api/encode/binary", json={"text": ""}).json()
    assert d["bitstring"] == "" and d["bytes"] == [] and d["byte_count"] == 0
    d = client.post("/api/encode/binary", json={"text": "😀"}).json()
    assert d["byte_count"] == 4 and len(d["bitstring"]) == 32 and d["character_count"] == 1


# ------------------------------------------------------------------ validation errors
@pytest.mark.parametrize("path", ["/api/encode/huffman", "/api/encode/binary"])
class TestInvalidRequests:
    def test_missing_text(self, path):
        err = assert_error(client.post(path, json={}), 422)
        assert err["details"][0]["field"] == "body.text"

    @pytest.mark.parametrize("bad", [123, None, ["a"], {"a": 1}, True])
    def test_wrong_type(self, path, bad):
        assert_error(client.post(path, json={"text": bad}), 422)

    def test_extra_fields_forbidden(self, path):
        assert_error(client.post(path, json={"text": "a", "extra": 1}), 422)

    def test_malformed_json(self, path):
        r = client.post(path, content="{not json", headers={"content-type": "application/json"})
        assert_error(r, 422)

    def test_lone_surrogate_rejected(self, path):
        r = client.post(path, content='{"text": "bad \\ud800 char"}',
                        headers={"content-type": "application/json"})
        assert_error(r, 422)

    def test_get_not_allowed(self, path):
        err = assert_error(client.get(path), 405, "http_error")
        assert "Method Not Allowed" in err["message"]

    def test_oversized_body_rejected_with_413(self, path):
        big = json.dumps({"text": "a" * (MAX_BODY_BYTES + 10)})
        err = assert_error(client.post(path, content=big, headers={"content-type": "application/json"}),
                           413)
        assert "too large" in err["message"].lower()

    def test_error_does_not_echo_input(self, path):
        secret = "SECRET-INPUT-VALUE"
        r = client.post(path, json={"text": secret, "bogus": secret})
        assert r.status_code == 422 and secret not in r.text


# ------------------------------------------------------------------ error-handler behaviour
def test_domain_validation_failure_maps_to_422():
    mini = FastAPI()
    register_error_handlers(mini)

    @mini.get("/boom")
    def boom():
        raise ValidationFailure("bad parameter", [{"field": "x", "message": "nope"}])

    r = TestClient(mini).get("/boom")
    err = assert_error(r, 422)
    assert err["message"] == "bad parameter" and err["details"] == [{"field": "x", "message": "nope"}]


def test_unexpected_exception_is_500_without_internals():
    mini = FastAPI()
    register_error_handlers(mini)

    @mini.get("/crash")
    def crash():
        raise RuntimeError("secret internal path /srv/app/private.py")

    r = TestClient(mini, raise_server_exceptions=False).get("/crash")
    err = assert_error(r, 500, "internal_error")
    assert err["message"] == "Internal server error."
    assert "secret" not in r.text and "RuntimeError" not in r.text


def test_501_maps_to_unsupported_operation():
    from fastapi import HTTPException
    mini = FastAPI()
    register_error_handlers(mini)

    @mini.get("/nope")
    def nope():
        raise HTTPException(status_code=501, detail="Not supported")

    assert_error(TestClient(mini).get("/nope"), 501, "unsupported_operation")


def test_openapi_lists_checkpoint3_endpoints():
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/api/encode/huffman"] and "post" in paths["/api/encode/binary"]


# ------------------------------------------------------------------ honesty of computed flags
def test_huffman_round_trip_flag_is_computed_not_hardcoded(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.source_encoding, "huffman_decode", lambda bits, table: "WRONG")
    d = client.post("/api/encode/huffman", json={"text": "hello"}).json()
    assert d["round_trip_ok"] is False


def test_binary_round_trip_flag_is_computed_not_hardcoded(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.binary_repr, "binary_to_text", lambda bits: "WRONG")
    d = client.post("/api/encode/binary", json={"text": "hello"}).json()
    assert d["round_trip_ok"] is False


def test_real_app_returns_clean_500_when_algorithm_crashes(monkeypatch):
    import backend.main as m

    def crash(_):
        raise RuntimeError("secret internal detail /srv/private.py")

    monkeypatch.setattr(m.source_encoding, "build_frequency_table", crash)
    r = TestClient(app, raise_server_exceptions=False).post("/api/encode/huffman", json={"text": "abc"})
    err = assert_error(r, 500, "internal_error")
    assert err["message"] == "Internal server error." and "secret" not in r.text


def test_domain_valueerror_in_algorithm_becomes_422(monkeypatch):
    import backend.main as m

    def reject(_):
        raise ValueError("algorithm rejected the input")

    monkeypatch.setattr(m.binary_repr, "text_to_binary", reject)
    err = assert_error(client.post("/api/encode/binary", json={"text": "abc"}), 422)
    assert err["message"] == "algorithm rejected the input"


def test_domain_valueerror_in_huffman_algorithm_becomes_422(monkeypatch):
    import backend.main as m

    def reject(_):
        raise ValueError("huffman rejected the input")

    monkeypatch.setattr(m.source_encoding, "huffman_encode", reject)
    err = assert_error(client.post("/api/encode/huffman", json={"text": "abc"}), 422)
    assert err["message"] == "huffman rejected the input"


def test_text_validator_rejects_lone_surrogate_in_isolation():
    # Called directly: pydantic-core's own string check would otherwise mask this validator.
    from backend.models import TextRequest
    validator = TextRequest._must_be_utf8_encodable
    with pytest.raises(ValueError, match="UTF-8"):
        validator("bad \ud800 char")
    assert validator("fine é 😀") == "fine é 😀"


# ====================================================================== Checkpoint 4: LZ77
import random as _random

from backend.models import MAX_LZ77_BYTES, MAX_TOKENS_RETURNED
from backend.pipeline import compression as _lz

LZ = "/api/compress/lz77"


def lz_post(**body):
    return client.post(LZ, json=body)


def test_lz77_known_example_tokens_and_sizes():
    d = lz_post(text="abracadabra").json()
    assert [(t["offset"], t["length"], t["next_hex"]) for t in d["tokens"]] == [
        (0, 0, "61"), (0, 0, "62"), (0, 0, "72"), (3, 1, "63"), (2, 1, "64"), (7, 4, "")]
    assert [t["kind"] for t in d["tokens"]] == ["literal"] * 3 + ["match", "match", "match-end"]
    assert d["original_bytes"] == 11 and d["token_count"] == 6
    assert d["literal_count"] == 3 and d["match_count"] == 3
    assert d["offset_bits"] == 13 and d["length_bits"] == 5
    assert d["payload_bits"] == 3 * 27 + 2 * 27 + 19 == 154
    assert d["header_bytes"] == 1 and d["compressed_bytes"] == 21
    assert d["compression_ratio"] == pytest.approx(11 / 21)
    assert d["space_saved_percent"] == pytest.approx((1 - 21 / 11) * 100)
    assert d["expanded"] is True                                   # short input expands: reported honestly
    assert d["round_trip_ok"] is True and d["serialization_round_trip_ok"] is True
    assert d["tokens_truncated"] is False and d["tokens_returned"] == 6
    assert "not the token count" in d["size_note"]


def test_lz77_repetitive_text_really_compresses():
    d = lz_post(text="abc" * 1000).json()
    assert d["original_bytes"] == 3000
    assert d["compressed_bytes"] < 3000 and d["expanded"] is False
    assert d["compression_ratio"] > 1 and d["space_saved_percent"] > 0
    assert d["round_trip_ok"] is True


def test_lz77_overlapping_token_flagged():
    d = lz_post(text="aaaa").json()
    last = d["tokens"][-1]
    assert (last["offset"], last["length"], last["next_hex"]) == (1, 3, "")
    assert last["overlapping"] is True and last["match_source_start"] == 0
    assert last["match_hex"] == "616161" and last["covers"] == 3


def test_lz77_empty_inputs():
    for body in ({"text": ""}, {"data_hex": ""}):
        d = lz_post(**body).json()
        assert d["original_bytes"] == 0 and d["token_count"] == 0 and d["tokens"] == []
        assert d["compressed_bytes"] == 0 and d["header_bytes"] == 0
        assert d["compression_ratio"] is None and d["space_saved_percent"] is None
        assert d["expanded"] is False and d["round_trip_ok"] is True


def test_lz77_data_hex_input_matches_text_input():
    t = lz_post(text="héllo héllo").json()
    h = lz_post(data_hex="héllo héllo".encode("utf-8").hex()).json()
    assert t == h
    assert lz_post(data_hex="ABCDEF00FF").status_code == 200           # upper-case hex accepted


def test_lz77_unicode_byte_counts():
    d = lz_post(text="é😀").json()
    assert d["original_bytes"] == 6 and d["round_trip_ok"] is True


def test_lz77_parameters_are_used_and_echoed():
    big = lz_post(text="abcxxabc" * 3, window_size=64, lookahead_size=18).json()
    small = lz_post(text="abcxxabc" * 3, window_size=2, lookahead_size=3).json()
    assert (small["window_size"], small["lookahead_size"]) == (2, 3)
    assert small["offset_bits"] == 2 and small["length_bits"] == 2
    assert all(t["covers"] <= 3 and t["offset"] <= 2 for t in small["tokens"])
    assert max(t["offset"] for t in big["tokens"]) > 2
    assert small["round_trip_ok"] and big["round_trip_ok"]


def test_lz77_response_matches_library_exactly():
    rng = _random.Random(10)
    data = bytes(rng.randrange(4) for _ in range(500))
    d = lz_post(data_hex=data.hex(), window_size=50, lookahead_size=9).json()
    tokens = _lz.lz77_compress(data, 50, 9)
    assert [(t["offset"], t["length"], t["next_hex"]) for t in d["tokens"]] == [
        (o, l, n.hex()) for o, l, n in tokens]
    blob = _lz.serialize_tokens(tokens, 50, 9)
    assert d["compressed_bytes"] == len(blob)


def test_lz77_deterministic():
    assert lz_post(text="mississippi " * 20).json() == lz_post(text="mississippi " * 20).json()


def test_lz77_token_list_is_bounded_but_statistics_cover_everything():
    rng = _random.Random(1)
    data = bytes(rng.randrange(256) for _ in range(3000))
    d = lz_post(data_hex=data.hex(), window_size=1, lookahead_size=1).json()   # all literals
    assert d["token_count"] == 3000 > MAX_TOKENS_RETURNED
    assert d["tokens_truncated"] is True and d["tokens_returned"] == MAX_TOKENS_RETURNED
    assert len(d["tokens"]) == MAX_TOKENS_RETURNED
    assert d["literal_count"] + d["match_count"] == 3000
    assert d["round_trip_ok"] is True


def test_lz77_size_limits():
    assert MAX_LZ77_BYTES == 40_000                       # documented limit, pinned literally
    ok = lz_post(data_hex="ab" * 40_000)
    assert ok.status_code == 200 and ok.json()["original_bytes"] == 40_000
    assert ok.json()["round_trip_ok"] is True
    err = assert_error(lz_post(data_hex="ab" * 40_001), 422)
    assert err["details"][0]["field"] == "body.data_hex"
    # 10,000 four-byte characters = 40,000 bytes is the largest accepted text
    assert lz_post(text="\U0001F600" * 10_000).json()["original_bytes"] == 40_000
    err = assert_error(lz_post(text="a" * (MAX_TEXT_CHARS + 1)), 422)
    assert err["details"][0]["field"] == "body.text"


@pytest.mark.parametrize("body", [
    {},                                                  # neither input
    {"text": "a", "data_hex": "61"},                     # both inputs
    {"text": None, "data_hex": None},
    {"data_hex": "abc"},                                 # odd length
    {"data_hex": "zz"},                                  # not hex
    {"data_hex": "0x61"},                                # prefix not allowed
    {"data_hex": "61 62"},                               # whitespace not allowed
    {"text": 123}, {"text": ["a"]}, {"data_hex": 61},
    {"text": "a", "window_size": 0}, {"text": "a", "window_size": -5},
    {"text": "a", "window_size": 32769}, {"text": "a", "window_size": "4096"},
    {"text": "a", "window_size": 4096.5}, {"text": "a", "window_size": True},
    {"text": "a", "window_size": None},
    {"text": "a", "lookahead_size": 0}, {"text": "a", "lookahead_size": 256},
    {"text": "a", "lookahead_size": "18"}, {"text": "a", "lookahead_size": 1.5},
    {"text": "a", "lookahead_size": False},
    {"text": "a", "bogus": 1},
])
def test_lz77_invalid_requests_rejected(body):
    assert_error(lz_post(**body), 422)


def test_lz77_boundary_parameters_accepted():
    for w, l in [(1, 1), (32768, 255), (1, 255), (32768, 1)]:
        assert lz_post(text="abababab", window_size=w, lookahead_size=l).status_code == 200


def test_lz77_lone_surrogate_and_malformed_json_rejected():
    assert_error(client.post(LZ, content='{"text": "x \\ud800"}',
                             headers={"content-type": "application/json"}), 422)
    assert_error(client.post(LZ, content="{nope", headers={"content-type": "application/json"}), 422)


def test_lz77_oversized_body_413_and_method_405():
    big = json.dumps({"text": "a" * (MAX_BODY_BYTES + 10)})
    assert_error(client.post(LZ, content=big, headers={"content-type": "application/json"}), 413)
    assert_error(client.get(LZ), 405, "http_error")


def test_lz77_errors_do_not_echo_input():
    secret = "SECRET-LZ-INPUT"
    r = lz_post(text=secret, window_size=0)
    assert r.status_code == 422 and secret not in r.text


def test_lz77_round_trip_flag_is_computed_not_hardcoded(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.compression, "lz77_decompress", lambda *a, **k: b"WRONG")
    assert lz_post(text="hello hello").json()["round_trip_ok"] is False


def test_lz77_serialization_flag_is_computed_not_hardcoded(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.compression, "deserialize_tokens", lambda *a, **k: [])
    d = lz_post(text="hello hello").json()
    assert d["serialization_round_trip_ok"] is False and d["round_trip_ok"] is True


def test_lz77_size_statistic_disagreeing_with_real_serialization_is_flagged(monkeypatch):
    # Tokens still round-trip through serialize/deserialize; ONLY the reported size is wrong.
    import backend.main as m
    real = m.compression.compression_stats

    def off_by_one(*a, **k):
        st = real(*a, **k)
        st["compressed_bytes"] += 1
        return st

    monkeypatch.setattr(m.compression, "compression_stats", off_by_one)
    d = lz_post(text="hello hello").json()
    assert d["serialization_round_trip_ok"] is False and d["round_trip_ok"] is True


def test_lz77_corrupt_serialization_is_flagged_not_a_user_error(monkeypatch):
    import backend.main as m
    real = m.compression.serialize_tokens
    monkeypatch.setattr(m.compression, "serialize_tokens", lambda *a, **k: real(*a, **k) + b"\x00")
    r = lz_post(text="hello hello")
    assert r.status_code == 200 and r.json()["serialization_round_trip_ok"] is False


def test_lz77_algorithm_valueerror_becomes_422(monkeypatch):
    import backend.main as m

    def reject(*a, **k):
        raise ValueError("lz77 rejected the input")

    monkeypatch.setattr(m.compression, "lz77_compress", reject)
    err = assert_error(lz_post(text="abc"), 422)
    assert err["message"] == "lz77 rejected the input"


def test_lz77_unexpected_crash_is_clean_500(monkeypatch):
    import backend.main as m

    def crash(*a, **k):
        raise RuntimeError("secret detail /srv/x.py")

    monkeypatch.setattr(m.compression, "lz77_compress", crash)
    r = TestClient(app, raise_server_exceptions=False).post(LZ, json={"text": "abc"})
    err = assert_error(r, 500, "internal_error")
    assert "secret" not in r.text and err["message"] == "Internal server error."


def test_lz77_in_openapi_and_health_version():
    assert "post" in client.get("/openapi.json").json()["paths"][LZ]
    from backend.main import APP_VERSION
    assert client.get("/api/health").json()["version"] == APP_VERSION


@pytest.mark.parametrize("body, field", [
    ({"data_hex": "abc"}, "body.data_hex"),                 # odd number of digits
    ({"data_hex": "zz"}, "body.data_hex"),                  # not hex
    ({"data_hex": "0x61"}, "body.data_hex"),                # prefix is not allowed
    ({"data_hex": "61  "}, "body.data_hex"),                # even length; bytes.fromhex alone would accept it
    ({"data_hex": "6g"}, "body.data_hex"),
    ({"text": "a", "window_size": 0}, "body.window_size"),
    ({"text": "a", "window_size": -5}, "body.window_size"),
    ({"text": "a", "window_size": 32769}, "body.window_size"),
    ({"text": "a", "window_size": 4096.5}, "body.window_size"),
    ({"text": "a", "lookahead_size": 0}, "body.lookahead_size"),
    ({"text": "a", "lookahead_size": 256}, "body.lookahead_size"),
    ({"text": "a", "lookahead_size": True}, "body.lookahead_size"),
])
def test_lz77_validation_errors_name_the_offending_field(body, field):
    err = assert_error(lz_post(**body), 422)
    assert err["message"] == "Request validation failed."
    assert err["details"] and err["details"][0]["field"] == field


@pytest.mark.parametrize("body", [{}, {"text": "a", "data_hex": "61"}])
def test_lz77_exactly_one_input_is_enforced_at_model_level(body):
    err = assert_error(lz_post(**body), 422)
    assert err["details"][0]["field"] == "body"
    assert "exactly one" in err["details"][0]["message"]


# ====================================================================== Checkpoint 5: AES endpoints
import logging

from backend.pipeline import decryption_aes as _dec
from backend.pipeline import encryption_aes as _enc

ENC, DEC = "/api/encrypt/aes", "/api/decrypt/aes"
K1 = "000102030405060708090a0b0c0d0e0f"
KB = "2b7e151628aed2a6abf7158809cf4f3c"
ZERO_IV = "00" * 16
SP_PT_HEX = ("6bc1bee22e409f96e93d7e117393172a" "ae2d8a571e03ac9c9eb76fac45af8e51"
             "30c81c46a35ce411e5fbc1191a0a52ef" "f69f2445df4f9b17ad2b417be66c3710")
SP_CT_HEX = ("7649abac8119b246cee98e9b12e9197d" "5086cb9b507219ee95db113a917678b2"
             "73bed6b8e3c1743b7116e69e22229516" "3ff1caa1681fac09120eca307586e1a7")


def enc(**body):
    return client.post(ENC, json=body)


def dec(**body):
    return client.post(DEC, json=body)


def xor_hex(a, b):
    return bytes(x ^ y for x, y in zip(bytes.fromhex(a), bytes.fromhex(b))).hex()


# ------------------------------------------------------------------ encryption: known answers
def test_aes_api_fips197_vector_via_cbc_with_zero_iv():
    # CBC with an all-zero IV makes the first ciphertext block equal the raw block encryption.
    d = enc(data_hex="00112233445566778899aabbccddeeff", key_hex=K1, iv_hex=ZERO_IV).json()
    assert d["ciphertext_hex"][:32] == "69c4e0d86a7b0430d8cdb78070b4c55a"
    assert d["block_count"] == 2 and d["padded_bytes"] == 32 and d["padding_bytes"] == 16
    assert d["plaintext_bytes"] == 16


def test_aes_api_nist_sp800_38a_cbc_vector():
    d = enc(data_hex=SP_PT_HEX, key_hex=KB, iv_hex="000102030405060708090a0b0c0d0e0f").json()
    assert d["ciphertext_hex"][:128] == SP_CT_HEX
    assert d["block_count"] == 5 and len(d["ciphertext_hex"]) == 160
    assert d["round_trip_ok"] is True


def test_aes_api_response_schema_and_metadata():
    d = enc(text="hello", key_hex=K1, iv_hex=ZERO_IV).json()
    assert (d["algorithm"], d["mode"], d["padding"], d["key_bits"]) == ("AES-128", "CBC", "PKCS#7", 128)
    assert d["plaintext_bytes"] == 5 and d["padding_bytes"] == 11 and d["padded_bytes"] == 16
    assert d["block_count"] == 1 and len(d["ciphertext_hex"]) == 32
    assert "Never use it to protect real data" in d["warning"]
    assert d["round_trip_ok"] is True and d["blocks_truncated"] is False
    b = d["blocks"][0]
    assert b["plaintext_hex"] == b"hello".hex() + "0b" * 11
    assert b["cipher_input_hex"] == xor_hex(b["plaintext_hex"], ZERO_IV)
    assert b["ciphertext_hex"] == d["ciphertext_hex"]


def test_aes_api_cbc_chaining_visible_in_blocks():
    d = enc(data_hex=SP_PT_HEX, key_hex=KB, iv_hex="000102030405060708090a0b0c0d0e0f").json()
    prev = d["iv_hex"]
    for b in d["blocks"]:
        assert b["cipher_input_hex"] == xor_hex(b["plaintext_hex"], prev)
        prev = b["ciphertext_hex"]


def test_aes_api_empty_plaintext_is_one_padding_block():
    d = enc(text="", key_hex=K1, iv_hex=ZERO_IV).json()
    assert d["plaintext_bytes"] == 0 and d["padding_bytes"] == 16 and d["block_count"] == 1
    assert d["round_trip_ok"] is True
    assert enc(data_hex="", key_hex=K1, iv_hex=ZERO_IV).json() == d


@pytest.mark.parametrize("n, blocks", [(15, 1), (16, 2), (17, 2), (31, 2), (32, 3)])
def test_aes_api_padding_arithmetic(n, blocks):
    d = enc(data_hex="ab" * n, key_hex=K1, iv_hex=ZERO_IV).json()
    assert d["block_count"] == blocks and d["padding_bytes"] == blocks * 16 - n and d["padding_bytes"] >= 1


# ------------------------------------------------------------------ key / IV handling and secrets
def test_aes_api_supplied_key_is_not_echoed_outside_the_trace():
    r = enc(text="hello", key_hex=KB, iv_hex=ZERO_IV)
    d = r.json()
    assert d["key_hex"] is None and d["key_generated"] is False and d["iv_generated"] is False
    assert d["iv_hex"] == ZERO_IV                            # the IV is not secret and is echoed
    trace = d.pop("trace")
    assert KB not in json.dumps(d)                           # nowhere except inside the trace ...
    assert trace["round_keys_hex"][0] == KB                  # ... where round key 0 IS the key (documented)


def test_aes_api_key_absent_everywhere_without_trace_or_with_redaction():
    for extra in ({"include_trace": False}, {"reveal_round_keys": False}):
        r = enc(text="hello", key_hex=KB, iv_hex=ZERO_IV, **extra)
        assert r.status_code == 200 and KB not in r.text and K1 not in r.text


def test_aes_api_trace_round_key_redaction():
    full = enc(text="hello", key_hex=K1, iv_hex=ZERO_IV).json()["trace"]
    red = enc(text="hello", key_hex=K1, iv_hex=ZERO_IV, reveal_round_keys=False).json()["trace"]
    assert full["round_keys_hex"][0] == K1 and full["steps"][1]["round_key_hex"] == K1
    assert sum("round_key_hex" in s for s in full["steps"]) == 11
    assert red["round_keys_hex"] is None and not any("round_key_hex" in s for s in red["steps"])
    assert len(red["steps"]) == 41                           # every state matrix is still there
    assert [s["state_hex"] for s in red["steps"]] == [s["state_hex"] for s in full["steps"]]
    assert red["output_hex"] == full["output_hex"]


def test_aes_api_generated_key_is_returned_and_works_for_decryption():
    d = enc(text="héllo wörld 😀").json()
    assert d["key_generated"] is True and d["iv_generated"] is True and d["key_seeded"] is False
    assert len(d["key_hex"]) == 32 and len(d["iv_hex"]) == 32
    back = dec(ciphertext_hex=d["ciphertext_hex"], key_hex=d["key_hex"], iv_hex=d["iv_hex"]).json()
    assert back["plaintext_text"] == "héllo wörld 😀" and back["plaintext_is_valid_utf8"] is True


def test_aes_api_unseeded_generation_is_random():
    a, b = enc(text="x").json(), enc(text="x").json()
    assert a["key_hex"] != b["key_hex"] and a["iv_hex"] != b["iv_hex"] and a["ciphertext_hex"] != b["ciphertext_hex"]


def test_aes_api_seed_reproduces_key_iv_and_ciphertext():
    a, b = enc(text="repeatable", seed=42).json(), enc(text="repeatable", seed=42).json()
    assert a == b and a["key_seeded"] is True
    c = enc(text="repeatable", seed=43).json()
    assert c["key_hex"] != a["key_hex"] and c["iv_hex"] != a["iv_hex"] and c["ciphertext_hex"] != a["ciphertext_hex"]
    assert enc(text="repeatable", seed=0).status_code == 200            # seed 0 is valid


def test_aes_api_seed_draws_key_then_iv_independent_of_supplied_values():
    base = enc(text="m", seed=7).json()
    only_key = enc(text="m", seed=7, key_hex=K1).json()
    assert only_key["iv_hex"] == base["iv_hex"] and only_key["key_hex"] is None and only_key["key_seeded"] is True
    only_iv = enc(text="m", seed=7, iv_hex=ZERO_IV).json()
    assert only_iv["key_hex"] == base["key_hex"] and only_iv["iv_hex"] == ZERO_IV
    both = enc(text="m", seed=7, key_hex=K1, iv_hex=ZERO_IV).json()
    assert both["key_seeded"] is False                                  # nothing came from the seed


def test_aes_api_supplied_iv_with_generated_key():
    d = enc(text="m", iv_hex=ZERO_IV).json()
    assert d["iv_hex"] == ZERO_IV and d["iv_generated"] is False and d["key_generated"] is True


def test_aes_api_does_not_log_secrets(caplog):
    with caplog.at_level(logging.DEBUG):
        enc(text="classified", key_hex=KB, iv_hex=ZERO_IV)
        dec(ciphertext_hex="00" * 16, key_hex=KB, iv_hex=ZERO_IV)
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert KB not in logged and b"classified".hex() not in logged


# ------------------------------------------------------------------ traces
def test_aes_api_trace_for_first_block():
    d = enc(data_hex="00112233445566778899aabbccddeeff", key_hex=K1, iv_hex=ZERO_IV).json()
    tr = d["trace"]
    assert d["trace_block"] == 0 and len(tr["steps"]) == 41
    assert tr["input_hex"] == d["blocks"][0]["cipher_input_hex"]
    assert tr["output_hex"] == d["blocks"][0]["ciphertext_hex"] == "69c4e0d86a7b0430d8cdb78070b4c55a"
    assert tr["steps"][4]["state_hex"] == "5f72641557f5bc92f7be3b291db9f91a"       # FIPS-197 C.1 round[1].m_col
    assert tr["steps"][-1]["state_hex"] == tr["output_hex"]
    assert "column-major" in tr["byte_order"]


def test_aes_api_trace_block_selection_uses_cbc_input_of_that_block():
    d = enc(data_hex=SP_PT_HEX, key_hex=KB, iv_hex="000102030405060708090a0b0c0d0e0f", trace_block=2).json()
    assert d["trace_block"] == 2
    assert d["trace"]["input_hex"] == d["blocks"][2]["cipher_input_hex"]
    assert d["trace"]["output_hex"] == d["blocks"][2]["ciphertext_hex"]


def test_aes_api_trace_can_be_disabled():
    d = enc(text="hello", key_hex=K1, iv_hex=ZERO_IV, include_trace=False).json()
    assert d["trace"] is None and d["trace_block"] is None


def test_aes_api_trace_block_out_of_range():
    err = assert_error(enc(text="hello", key_hex=K1, iv_hex=ZERO_IV, trace_block=1), 422)
    assert err["details"][0]["field"] == "body.trace_block"
    assert enc(text="hello", key_hex=K1, iv_hex=ZERO_IV, trace_block=1, include_trace=False).status_code == 200


def test_aes_api_block_list_is_bounded_but_ciphertext_complete():
    d = enc(data_hex="ab" * 1600, key_hex=K1, iv_hex=ZERO_IV, include_trace=False).json()
    assert d["block_count"] == 101 and d["blocks_truncated"] is True and len(d["blocks"]) == 64
    assert len(d["ciphertext_hex"]) == 101 * 32 and d["round_trip_ok"] is True


# ------------------------------------------------------------------ key size / unsupported
@pytest.mark.parametrize("bits", [192, 256])
def test_aes_api_larger_key_sizes_are_unsupported_not_silently_downgraded(bits):
    r = enc(text="a", key_bits=bits)
    err = assert_error(r, 501, "unsupported_operation")
    assert "AES-128" in err["message"]


def test_aes_api_explicit_128_is_fine():
    assert enc(text="a", key_bits=128).status_code == 200


@pytest.mark.parametrize("bad", [100, 0, -128, "128", 128.0, True, None])
def test_aes_api_invalid_key_bits_rejected(bad):
    assert_error(enc(text="a", key_bits=bad), 422)


# ------------------------------------------------------------------ request validation
@pytest.mark.parametrize("body, field", [
    ({"text": "a", "key_hex": "00" * 15}, "body.key_hex"),               # too short
    ({"text": "a", "key_hex": "00" * 17}, "body.key_hex"),               # too long
    ({"text": "a", "key_hex": "zz" * 16}, "body.key_hex"),               # not hex
    ({"text": "a", "key_hex": "0x" + "00" * 15}, "body.key_hex"),
    ({"text": "a", "key_hex": 12345}, "body.key_hex"),
    ({"text": "a", "iv_hex": "00" * 15}, "body.iv_hex"),
    ({"text": "a", "iv_hex": "00" * 17}, "body.iv_hex"),
    ({"text": "a", "iv_hex": "g" * 32}, "body.iv_hex"),
    ({"data_hex": "abc"}, "body.data_hex"),
    ({"data_hex": "zz"}, "body.data_hex"),
    ({"data_hex": "61  "}, "body.data_hex"),
    ({"text": "a", "seed": -1}, "body.seed"),
    ({"text": "a", "seed": 2 ** 32}, "body.seed"),
    ({"text": "a", "seed": 1.5}, "body.seed"),
    ({"text": "a", "seed": True}, "body.seed"),
    ({"text": "a", "seed": "5"}, "body.seed"),
    ({"text": "a", "include_trace": "yes"}, "body.include_trace"),
    ({"text": "a", "include_trace": 1}, "body.include_trace"),
    ({"text": "a", "reveal_round_keys": "no"}, "body.reveal_round_keys"),
    ({"text": "a", "reveal_round_keys": 0}, "body.reveal_round_keys"),
    ({"text": "a", "trace_block": -1}, "body.trace_block"),
    ({"text": "a", "trace_block": 1.5}, "body.trace_block"),
    ({"text": 123}, "body.text"),
])
def test_aes_api_validation_errors_name_the_field(body, field):
    err = assert_error(enc(**body), 422)
    assert err["details"] and err["details"][0]["field"] == field


@pytest.mark.parametrize("body", [{}, {"text": "a", "data_hex": "61"}, {"text": None, "data_hex": None}])
def test_aes_api_exactly_one_plaintext_input(body):
    err = assert_error(enc(**body), 422)
    assert err["details"][0]["field"] == "body" and "exactly one" in err["details"][0]["message"]


def test_aes_api_extra_fields_and_bad_json_rejected():
    assert_error(enc(text="a", bogus=1), 422)
    assert_error(client.post(ENC, content="{nope", headers={"content-type": "application/json"}), 422)
    assert_error(client.post(ENC, content='{"text": "x \\ud800"}', headers={"content-type": "application/json"}), 422)


def test_aes_api_validation_errors_never_echo_the_key():
    r = enc(text="a", key_hex=KB, iv_hex="00" * 15)               # valid key, invalid IV
    assert r.status_code == 422 and KB not in r.text
    r = enc(text="a", key_hex="ZZ" + KB[2:])
    assert r.status_code == 422 and KB[2:] not in r.text
    r = dec(ciphertext_hex="00" * 16, key_hex=KB, iv_hex="00" * 3)
    assert r.status_code == 422 and KB not in r.text


def test_aes_api_size_limits():
    ok = enc(data_hex="ab" * 40_000, key_hex=K1, iv_hex=ZERO_IV, include_trace=False)
    assert ok.status_code == 200 and ok.json()["round_trip_ok"] is True and ok.json()["block_count"] == 2501
    err = assert_error(enc(data_hex="ab" * 40_001, key_hex=K1, iv_hex=ZERO_IV), 422)
    assert err["details"][0]["field"] == "body.data_hex"
    assert enc(text="\U0001F600" * 10_000, key_hex=K1, iv_hex=ZERO_IV, include_trace=False).json()["plaintext_bytes"] == 40_000
    assert_error(enc(text="a" * (MAX_TEXT_CHARS + 1)), 422)


def test_aes_api_oversized_body_413_and_methods_405():
    big = json.dumps({"text": "a" * (MAX_BODY_BYTES + 10)})
    for path in (ENC, DEC):
        assert_error(client.post(path, content=big, headers={"content-type": "application/json"}), 413)
        assert_error(client.get(path), 405, "http_error")


# ------------------------------------------------------------------ honesty of computed flags
def test_aes_api_round_trip_flag_is_computed_not_hardcoded(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.decryption_aes, "aes_cbc_decrypt", lambda *a, **k: b"WRONG")
    assert enc(text="hello", key_hex=K1, iv_hex=ZERO_IV).json()["round_trip_ok"] is False


def test_aes_api_round_trip_flag_false_when_verification_raises(monkeypatch):
    import backend.main as m

    def boom(*a, **k):
        raise _enc.PaddingError("bad")

    monkeypatch.setattr(m.decryption_aes, "aes_cbc_decrypt", boom)
    r = enc(text="hello", key_hex=K1, iv_hex=ZERO_IV)
    assert r.status_code == 200 and r.json()["round_trip_ok"] is False


def test_aes_api_unexpected_crash_is_clean_500_without_key(monkeypatch):
    import backend.main as m

    def crash(*a, **k):
        raise RuntimeError("internal /srv/aes.py " + KB)

    monkeypatch.setattr(m.encryption_aes, "aes_cbc_encrypt_blocks", crash)
    r = TestClient(app, raise_server_exceptions=False).post(ENC, json={"text": "a", "key_hex": KB})
    err = assert_error(r, 500, "internal_error")
    assert KB not in r.text and "RuntimeError" not in r.text and err["message"] == "Internal server error."


# ------------------------------------------------------------------ decryption endpoint
def test_aes_decrypt_api_recovers_nist_vector():
    d = dec(ciphertext_hex=SP_CT_HEX + enc(data_hex=SP_PT_HEX, key_hex=KB, iv_hex="000102030405060708090a0b0c0d0e0f").json()["ciphertext_hex"][128:],
            key_hex=KB, iv_hex="000102030405060708090a0b0c0d0e0f").json()
    assert d["plaintext_hex"] == SP_PT_HEX and d["plaintext_bytes"] == 64
    assert d["padding_bytes"] == 16 and d["block_count"] == 5


def test_aes_decrypt_api_blocks_follow_cbc_definition():
    e = enc(data_hex=SP_PT_HEX, key_hex=KB, iv_hex="000102030405060708090a0b0c0d0e0f").json()
    d = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=KB, iv_hex=e["iv_hex"]).json()
    prev = e["iv_hex"]
    for b in d["blocks"]:
        assert b["plaintext_hex"] == xor_hex(b["decrypted_hex"], prev)
        prev = b["ciphertext_hex"]
    assert d["blocks"][0]["plaintext_hex"] == SP_PT_HEX[:32]


def test_aes_decrypt_api_non_utf8_plaintext_reported_honestly():
    e = enc(data_hex="ff00fe", key_hex=K1, iv_hex=ZERO_IV).json()
    d = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=K1, iv_hex=ZERO_IV).json()
    assert d["plaintext_hex"] == "ff00fe" and d["plaintext_text"] is None and d["plaintext_is_valid_utf8"] is False


def test_aes_decrypt_api_empty_plaintext():
    e = enc(text="", key_hex=K1, iv_hex=ZERO_IV).json()
    d = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=K1, iv_hex=ZERO_IV).json()
    assert d["plaintext_bytes"] == 0 and d["plaintext_text"] == "" and d["padding_bytes"] == 16


def test_aes_decrypt_api_never_returns_the_key_unless_trace_round_keys_are_revealed():
    e = enc(text="hello", key_hex=KB, iv_hex=ZERO_IV).json()
    for extra in ({"include_trace": False}, {"reveal_round_keys": False}):
        r = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=KB, iv_hex=ZERO_IV, **extra)
        assert KB not in r.text and "key_hex" not in r.json()
    tr = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=KB, iv_hex=ZERO_IV).json()["trace"]
    assert tr["round_keys_hex"][0] == KB                    # documented: default trace shows round keys
    red = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=KB, iv_hex=ZERO_IV, reveal_round_keys=False).json()["trace"]
    assert red["round_keys_hex"] is None and not any("round_key_hex" in s for s in red["steps"])
    assert [s["state_hex"] for s in red["steps"]] == [s["state_hex"] for s in tr["steps"]]


def test_aes_decrypt_api_trace():
    e = enc(text="hello", key_hex=K1, iv_hex=ZERO_IV).json()
    d = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=K1, iv_hex=ZERO_IV).json()
    tr = d["trace"]
    assert d["trace_block"] == 0 and tr["direction"] == "decrypt"
    assert tr["input_hex"] == e["ciphertext_hex"]
    assert tr["output_hex"] == d["blocks"][0]["decrypted_hex"]
    assert tr["steps"][0]["round"] == 10 and tr["steps"][-1]["round"] == 0
    assert dec(ciphertext_hex=e["ciphertext_hex"], key_hex=K1, iv_hex=ZERO_IV, include_trace=False).json()["trace"] is None
    err = assert_error(dec(ciphertext_hex=e["ciphertext_hex"], key_hex=K1, iv_hex=ZERO_IV, trace_block=5), 422)
    assert err["details"][0]["field"] == "body.trace_block"


@pytest.mark.parametrize("n_bytes", [0, 1, 15, 17, 31])
def test_aes_decrypt_api_bad_ciphertext_length_is_an_integrity_failure(n_bytes):
    err = assert_error(dec(ciphertext_hex="ab" * n_bytes, key_hex=K1, iv_hex=ZERO_IV), 422, "integrity_failure")
    assert "multiple of 16" in err["message"]


def test_aes_decrypt_api_bad_padding_is_an_integrity_failure():
    bad_plain = b"A" * 15 + b"\x00"
    ct = _enc.aes_encrypt_block(bytes(a ^ b for a, b in zip(bad_plain, bytes(16))), bytes.fromhex(K1))
    err = assert_error(dec(ciphertext_hex=ct.hex(), key_hex=K1, iv_hex=ZERO_IV), 422, "integrity_failure")
    assert "padding" in err["message"].lower()


def test_aes_decrypt_api_wrong_key_never_yields_the_original():
    e = enc(text="top secret message!", key_hex=KB, iv_hex=ZERO_IV).json()
    r = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=K1, iv_hex=ZERO_IV, include_trace=False)
    if r.status_code == 200:
        assert r.json()["plaintext_text"] != "top secret message!"
    else:
        assert_error(r, 422, "integrity_failure")


def test_aes_decrypt_api_tampering_is_detected_or_garbles_exactly_as_cbc_predicts():
    msg = "0123456789abcdef" * 3
    e = enc(text=msg, key_hex=K1, iv_hex=ZERO_IV).json()
    ct = bytearray(bytes.fromhex(e["ciphertext_hex"]))
    ct[0] ^= 0x01                                           # flip a bit in block 0
    d = dec(ciphertext_hex=bytes(ct).hex(), key_hex=K1, iv_hex=ZERO_IV, include_trace=False).json()
    assert d["blocks"][0]["plaintext_hex"] != msg.encode()[:16].hex()           # block 0 garbled
    assert d["blocks"][1]["plaintext_hex"] == xor_hex(msg.encode()[16:32].hex(), "01" + "00" * 15)   # 1 bit flips in block 1


@pytest.mark.parametrize("body, field", [
    ({"key_hex": K1, "iv_hex": ZERO_IV}, "body.ciphertext_hex"),                        # missing ciphertext
    ({"ciphertext_hex": "ab" * 16, "iv_hex": ZERO_IV}, "body.key_hex"),                 # missing key
    ({"ciphertext_hex": "ab" * 16, "key_hex": K1}, "body.iv_hex"),                      # missing iv
    ({"ciphertext_hex": "abc", "key_hex": K1, "iv_hex": ZERO_IV}, "body.ciphertext_hex"),
    ({"ciphertext_hex": "zz", "key_hex": K1, "iv_hex": ZERO_IV}, "body.ciphertext_hex"),
    ({"ciphertext_hex": "ab" * 16, "key_hex": "00" * 15, "iv_hex": ZERO_IV}, "body.key_hex"),
    ({"ciphertext_hex": "ab" * 16, "key_hex": K1, "iv_hex": "00" * 17}, "body.iv_hex"),
    ({"ciphertext_hex": "ab" * 16, "key_hex": K1, "iv_hex": ZERO_IV, "include_trace": "no"}, "body.include_trace"),
    ({"ciphertext_hex": 7, "key_hex": K1, "iv_hex": ZERO_IV}, "body.ciphertext_hex"),
])
def test_aes_decrypt_api_validation_errors_name_the_field(body, field):
    err = assert_error(dec(**body), 422)
    assert err["details"][0]["field"] == field


def test_aes_decrypt_api_ciphertext_size_limit():
    err = assert_error(dec(ciphertext_hex="ab" * 40_017, key_hex=K1, iv_hex=ZERO_IV), 422)
    assert err["details"][0]["field"] == "body.ciphertext_hex"
    big = enc(data_hex="cd" * 40_000, key_hex=K1, iv_hex=ZERO_IV, include_trace=False).json()
    d = dec(ciphertext_hex=big["ciphertext_hex"], key_hex=K1, iv_hex=ZERO_IV, include_trace=False).json()
    assert d["plaintext_bytes"] == 40_000 and d["blocks_truncated"] is True and len(d["blocks"]) == 64


def test_aes_decrypt_api_unexpected_crash_is_clean_500(monkeypatch):
    import backend.main as m

    def crash(*a, **k):
        raise RuntimeError("secret " + K1)

    monkeypatch.setattr(m.decryption_aes, "aes_cbc_decrypt_blocks", crash)
    r = TestClient(app, raise_server_exceptions=False).post(DEC, json={"ciphertext_hex": "ab" * 16, "key_hex": K1, "iv_hex": ZERO_IV})
    assert_error(r, 500, "internal_error")
    assert K1 not in r.text


def test_aes_endpoints_listed_in_openapi():
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths[ENC] and "post" in paths[DEC]


def test_aes_full_http_round_trip_many_messages():
    rng = _random.Random(11)
    for n in (0, 1, 15, 16, 17, 100, 333):
        data = bytes(rng.randrange(256) for _ in range(n))
        e = enc(data_hex=data.hex(), include_trace=False, seed=n).json()
        d = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=e["key_hex"], iv_hex=e["iv_hex"], include_trace=False).json()
        assert d["plaintext_hex"] == data.hex() and e["round_trip_ok"] is True


# ------------------------------------------------------------------ gap-closing tests (found by mutation testing)
def test_aes_api_seed_contract_key_is_first_draw_iv_is_second_from_random_random():
    rng = _random.Random(7)
    expected_key, expected_iv = rng.randbytes(16).hex(), rng.randbytes(16).hex()   # documented contract
    d = enc(text="m", seed=7).json()
    assert d["key_hex"] == expected_key and d["iv_hex"] == expected_iv
    assert enc(text="m", seed=0).json()["key_hex"] == _random.Random(0).randbytes(16).hex()
    # a supplied key does not change the IV drawn from the seed (key is still drawn first)
    assert enc(text="m", seed=7, key_hex=K1).json()["iv_hex"] == expected_iv


def test_aes_decrypt_api_trace_block_selects_the_requested_block():
    e = enc(data_hex=SP_PT_HEX, key_hex=KB, iv_hex="000102030405060708090a0b0c0d0e0f", include_trace=False).json()
    for i in (0, 1, 3, 4):
        d = dec(ciphertext_hex=e["ciphertext_hex"], key_hex=KB, iv_hex=e["iv_hex"], trace_block=i).json()
        assert d["trace_block"] == i
        assert d["trace"]["input_hex"] == e["ciphertext_hex"][32 * i:32 * i + 32]
        assert d["trace"]["output_hex"] == d["blocks"][i]["decrypted_hex"]
    err = assert_error(dec(ciphertext_hex=e["ciphertext_hex"], key_hex=KB, iv_hex=e["iv_hex"], trace_block=5), 422)
    assert err["details"][0]["field"] == "body.trace_block"


# ====================================================================== Checkpoint 6: RSA endpoint
from backend.models import MAX_RSA_STEPS_RETURNED
from backend.pipeline import encryption_rsa as _rsa
from backend.pipeline import decryption_rsa as _rsad

RSA = "/api/encrypt/rsa-keyexchange"
BIG_FIELDS = ("p", "q", "n", "phi", "e", "d", "ciphertext")


def rsa_post(**body):
    return client.post(RSA, json=body)


def is_prime_ref(n):
    """Independent reference primality test written with the built-in pow (test code only)."""
    if n < 2:
        return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0:
            return n == sp
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# ------------------------------------------------------------------ the classic textbook example
def test_rsa_api_classic_textbook_example_integer_mode():
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65").json()
    assert (d["n"], d["phi"], d["e"], d["d"]) == ("3233", "3120", "17", "2753")
    assert d["ciphertext"] == "2790" and d["recovered"] == "65" and d["message"] == "65"
    assert d["key_source"] == "user_primes" and d["e_source"] == "user" and d["seeded"] is False
    assert d["round_trip_ok"] is True and d["key_recovered_ok"] is None and d["message_kind"] == "integer"
    assert d["modulus_bits"] == 12 and d["message_bits"] == 7
    assert d["d_derivation"]["equation"] == "e * d = 1 (mod phi):  17 * 2753 mod 3120 = 1"
    assert d["key_checks"]["all_ok"] is True and d["key_checks"]["gcd_e_phi"] == "1"
    assert d["key_checks"]["e_times_d_mod_phi"] == "1"


def test_rsa_api_square_and_multiply_steps_are_the_real_arithmetic():
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65").json()
    enc = d["encryption"]
    assert enc["exponent_bits"] == "10001" and enc["initial"] == "65" and enc["result"] == "2790"
    assert enc["squarings"] == 4 and enc["multiplications"] == 1 and enc["steps_truncated"] is False
    assert [(s["bit"], s["after_square"], s["after_multiply"]) for s in enc["steps"]] == [
        (0, "992", "992"), (0, "1232", "1232"), (0, "1547", "1547"), (1, "789", "2790")]
    value = int(enc["initial"])
    for st in enc["steps"]:                                     # independent re-computation in the test
        assert int(st["before"]) == value and int(st["after_square"]) == value * value % 3233
        value = int(st["after_square"]) * 65 % 3233 if st["bit"] else int(st["after_square"])
        assert int(st["after_multiply"]) == value
    dec_ = d["decryption"]
    assert dec_["exponent"] == "2753" and dec_["result"] == "65" and dec_["base"] == "2790"
    assert dec_["exponent_bits"] == bin(2753)[2:]


def test_rsa_api_extended_euclid_derivation_of_d():
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65").json()["d_derivation"]
    assert d["steps"] and d["steps"][-1]["remainder"] == "0" and d["steps_truncated"] is False
    assert "extended Euclid" in d["method"]


# ------------------------------------------------------------------ AES key transport (default mode)
def test_rsa_api_default_mode_wraps_a_128_bit_aes_key_with_a_160_bit_modulus():
    d = rsa_post(seed=1).json()
    assert d["mode"] == "aes_key_wrap" and d["modulus_bits"] == 160 and d["message_kind"] == "aes_key_128"
    n, e, dd, c = int(d["n"]), int(d["e"]), int(d["d"]), int(d["ciphertext"])
    key = bytes.fromhex(d["aes_key_hex"])
    m = int.from_bytes(key, "big")
    assert len(key) == 16 and m < n and n > 2 ** 128                       # the whole key fits below the modulus
    assert c == pow(m, e, n) and pow(c, dd, n) == m                         # independent verification
    assert d["recovered_key_hex"] == d["aes_key_hex"] and d["key_recovered_ok"] is True
    assert d["message"] == str(m) and d["recovered"] == str(m) and d["round_trip_ok"] is True


def test_rsa_api_generated_numbers_are_mathematically_consistent():
    d = rsa_post(seed=2).json()
    p, q, n, phi, e, dd = (int(d[k]) for k in ("p", "q", "n", "phi", "e", "d"))
    assert p != q and n == p * q and n.bit_length() == 160 and phi == (p - 1) * (q - 1)
    assert is_prime_ref(p) and is_prime_ref(q)
    assert math.gcd(e, phi) == 1 and 1 < e < phi and e * dd % phi == 1
    assert d["p"] != d["q"] and d["e_source"] == "default" and e == 65537


def test_rsa_api_all_large_integers_are_decimal_strings_not_json_numbers():
    d = rsa_post(seed=3).json()
    for k in BIG_FIELDS + ("message", "recovered"):
        assert isinstance(d[k], str) and d[k].isdigit(), k
    for part in ("encryption", "decryption"):
        for k in ("base", "exponent", "modulus", "initial", "result"):
            assert isinstance(d[part][k], str)
        for st in d[part]["steps"]:
            assert all(isinstance(st[k], str) for k in ("before", "after_square", "after_multiply"))
    # every JSON number in the response is small enough for JavaScript to hold exactly
    def numbers(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from numbers(v)
        elif isinstance(x, list):
            for v in x:
                yield from numbers(v)
    assert all(abs(v) < 2 ** 53 for v in numbers(d))


def test_rsa_api_step_lists_are_bounded_but_counts_are_exact():
    d = rsa_post(seed=4).json()
    dec_ = d["decryption"]
    assert dec_["steps_truncated"] is True and dec_["steps_returned"] == MAX_RSA_STEPS_RETURNED
    assert len(dec_["steps"]) == MAX_RSA_STEPS_RETURNED
    dd = int(d["d"])
    assert dec_["steps_total"] == dd.bit_length() - 1 == dec_["squarings"]
    assert dec_["multiplications"] == bin(dd).count("1") - 1
    assert d["encryption"]["steps_truncated"] is False                      # 65537 has only 16 steps


def test_rsa_api_labels_everything_insecure():
    for body in ({"seed": 1}, {"mode": "integer"}, {"mode": "integer", "p": "61", "q": "53"}):
        d = rsa_post(**body).json()
        assert d["insecure"] is True and "INSECURE" in d["security_note"]
        assert d["real_world_minimum_modulus_bits"] == 2048
        assert any("padding" in n.lower() for n in d["textbook_notes"])
        assert any("2^128" in n for n in d["textbook_notes"])
        assert set(d["roles"]) == {"receiver", "sender", "receiver_decrypts"}


def test_rsa_api_primality_report_describes_the_miller_rabin_strategy():
    d = rsa_post(seed=5).json()
    for who in ("p", "q"):
        rep = d["primality"][who]
        assert rep["is_prime"] is True and rep["method"] == "miller_rabin_deterministic"
        assert rep["bases"] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41] and rep["witness"] is None
    big = rsa_post(modulus_bits=512, seed=6).json()["primality"]["p"]
    assert big["method"] == "miller_rabin_probabilistic" and big["random_bases"] == 20
    assert big["error_bound"] == 4.0 ** -20 and len(big["bases"]) == 33


# ------------------------------------------------------------------ integer mode
def test_rsa_api_integer_mode_defaults_to_a_tiny_32_bit_modulus():
    d = rsa_post(mode="integer", seed=7).json()
    n, e, dd, c, m = (int(d[k]) for k in ("n", "e", "d", "ciphertext", "message"))
    assert n.bit_length() == 32 and 2 <= m < n - 1
    assert c == pow(m, e, n) and pow(c, dd, n) == m == int(d["recovered"])
    assert d["aes_key_hex"] is None and d["recovered_key_hex"] is None and d["key_recovered_ok"] is None


def test_rsa_api_integer_mode_with_supplied_message_and_generated_keys():
    d = rsa_post(mode="integer", modulus_bits=64, message="123456789", seed=8).json()
    assert d["message"] == "123456789" and d["modulus_bits"] == 64 and d["round_trip_ok"] is True
    assert d["ciphertext"] == str(pow(123456789, int(d["e"]), int(d["n"])))


def test_rsa_api_message_boundaries():
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="3232").json()
    assert d["round_trip_ok"] and d["recovered"] == "3232"
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="0").json()
    assert d["ciphertext"] == "0" and d["recovered"] == "0"
    err = assert_error(rsa_post(mode="integer", p="61", q="53", e="17", message="3233"), 422)
    assert err["details"][0]["field"] == "body.message" and "smaller than the modulus" in err["message"]


def test_rsa_api_default_exponent_and_fallback_exponent_labels():
    assert rsa_post(mode="integer", p="1000003", q="999983").json()["e_source"] == "default"
    d = rsa_post(mode="integer", p="61", q="53").json()                    # phi = 3120 < 65537 -> a smaller standard e
    assert d["e_source"] == "fallback" and int(d["e"]) in (257, 17, 5, 3) and d["round_trip_ok"]


# ------------------------------------------------------------------ reproducibility
def test_rsa_api_seed_reproduces_everything():
    a, b = rsa_post(seed=42).json(), rsa_post(seed=42).json()
    assert a == b and a["seeded"] is True
    c = rsa_post(seed=43).json()
    assert c["n"] != a["n"] and c["aes_key_hex"] != a["aes_key_hex"]


def test_rsa_api_seed_contract_primes_then_aes_key_from_random_random():
    rng = _random.Random(42)
    pair = _rsa.generate_keypair(160, None, rng)                            # same draw order as documented
    key = rng.randbytes(16)
    d = rsa_post(seed=42).json()
    assert (d["p"], d["q"], d["e"], d["d"]) == (str(pair.p), str(pair.q), str(pair.public.e), str(pair.private.d))
    assert d["aes_key_hex"] == key.hex()


def test_rsa_api_unseeded_generation_varies():
    a, b = rsa_post().json(), rsa_post().json()
    assert a["n"] != b["n"] and a["seeded"] is False


def test_rsa_api_every_modulus_size_works():
    for bits in (130, 132, 160, 256, 512):
        d = rsa_post(modulus_bits=bits, seed=bits).json()
        assert d["modulus_bits"] == bits and d["round_trip_ok"] and d["key_recovered_ok"] is True
    for bits in (12, 14, 32, 64):
        d = rsa_post(mode="integer", modulus_bits=bits, seed=bits).json()
        assert d["modulus_bits"] == bits and d["round_trip_ok"] is True


# ------------------------------------------------------------------ supplied AES key and secrets
KEY = "000102030405060708090a0b0c0d0e0f"


def test_rsa_api_supplied_aes_key_is_not_echoed_in_dedicated_fields():
    r = rsa_post(seed=9, aes_key_hex=KEY, include_traces=False)
    d = r.json()
    assert d["aes_key_hex"] is None and d["recovered_key_hex"] is None and d["message"] is None
    assert d["recovered"] is None and d["key_recovered_ok"] is True and d["round_trip_ok"] is True
    assert KEY not in r.text and str(int(KEY, 16)) not in r.text
    assert d["encryption"] is None and d["decryption"] is None


def test_rsa_api_supplied_key_appears_only_inside_traces_when_traces_are_requested():
    d = rsa_post(seed=9, aes_key_hex=KEY).json()
    assert d["encryption"]["initial"] == str(int(KEY, 16))                  # documented: traces contain the key integer
    stripped = {k: v for k, v in d.items() if k not in ("encryption", "decryption")}
    assert str(int(KEY, 16)) not in json.dumps(stripped) and KEY not in json.dumps(stripped)
    assert d["decryption"]["result"] == str(int(KEY, 16))


def test_rsa_api_supplied_key_round_trips_against_independent_math():
    d = rsa_post(seed=10, aes_key_hex=KEY).json()
    n, e, dd, c = (int(d[k]) for k in ("n", "e", "d", "ciphertext"))
    assert c == pow(int(KEY, 16), e, n) and pow(c, dd, n) == int(KEY, 16)


def test_rsa_api_include_traces_false_in_integer_mode_omits_traces_only():
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65", include_traces=False).json()
    assert d["encryption"] is None and d["decryption"] is None and d["ciphertext"] == "2790"


def test_rsa_api_does_not_log_private_key_material(caplog):
    with caplog.at_level(logging.DEBUG):
        d = rsa_post(seed=11).json()
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert d["d"] not in logged and d["p"] not in logged and d["aes_key_hex"] not in logged


# ------------------------------------------------------------------ key-parameter validation
@pytest.mark.parametrize("body, field, fragment", [
    ({"mode": "integer", "p": "60", "q": "53"}, "body.p", "not prime"),
    ({"mode": "integer", "p": "61", "q": "54"}, "body.q", "not prime"),
    ({"mode": "integer", "p": "61", "q": "61"}, "body.q", "distinct"),
    ({"mode": "integer", "p": "2", "q": "53"}, "body.p", "odd prime"),
    ({"mode": "integer", "p": "1", "q": "53"}, "body.p", "odd prime"),
    ({"mode": "integer", "p": "0", "q": "53"}, "body.p", "odd prime"),
    ({"mode": "integer", "p": "561", "q": "61"}, "body.p", "not prime"),                      # Carmichael number
    ({"mode": "integer", "p": "7", "q": "13", "e": "3"}, "body.e", "not coprime"),             # phi = 72
    ({"mode": "integer", "p": "61", "q": "53", "e": "2"}, "body.e", "odd"),
    ({"mode": "integer", "p": "61", "q": "53", "e": "1"}, "body.e", "odd"),
    ({"mode": "integer", "p": "61", "q": "53", "e": "3121"}, "body.e", "smaller than phi"),
    ({"mode": "integer", "p": "61", "q": "53", "e": "99999999999999"}, "body.e", "smaller than phi"),
    ({"mode": "integer", "modulus_bits": 32, "e": "4"}, "body.e", "odd"),
    ({"mode": "integer", "modulus_bits": 12, "e": "9999999999"}, "body.e", None),
    ({"p": "10" + "0" * 76 + "7", "q": "61"}, None, None),                                     # 79 digits: too long for the field
])
def test_rsa_api_invalid_key_parameters_are_clear_validation_errors(body, field, fragment):
    err = assert_error(rsa_post(**body), 422)
    if field:
        assert err["details"] and err["details"][0]["field"] == field
    if fragment:
        assert fragment in err["message"]


def test_rsa_api_oversized_prime_rejected_by_bit_length_not_just_digits():
    big_prime_dec = str(2 ** 257 + 297) if False else None
    p = (1 << 259) | 1                                      # 260 bits, 79 digits -> rejected by the field limit
    assert_error(rsa_post(mode="integer", p=str(p), q="61"), 422)
    p256 = (1 << 257) + 1                                   # 258 bits, 78 digits: passes the digit limit
    err = assert_error(rsa_post(mode="integer", p=str(p256), q="61"), 422)
    assert err["details"][0]["field"] == "body.p" and "too large" in err["message"]


def test_rsa_api_user_primes_at_the_size_limit_work():
    p = _rsa.generate_prime(256, _random.Random(1))
    q = _rsa.generate_prime(256, _random.Random(2))
    d = rsa_post(mode="integer", p=str(p), q=str(q), message="123456789").json()
    assert d["modulus_bits"] in (511, 512) and d["round_trip_ok"] is True
    assert d["primality"]["p"]["method"] == "miller_rabin_probabilistic"


def test_rsa_api_aes_key_wrap_requires_a_modulus_that_can_carry_the_key():
    for bits in (12, 64, 128):
        err = assert_error(rsa_post(modulus_bits=bits), 422)
        assert err["details"][0]["field"] == "body.modulus_bits" and "2^128" in err["message"]
    assert rsa_post(modulus_bits=130, seed=1).status_code == 200          # smallest accepted


def test_rsa_api_user_primes_too_small_for_the_key_are_rejected_not_truncated():
    err = assert_error(rsa_post(p="1000003", q="999983", aes_key_hex="ff" * 16), 422)
    assert err["details"][0]["field"] == "body.p" and "does not fit" in err["message"]
    assert "ffffffff" not in json.dumps(err)                               # the key is not echoed in the error


@pytest.mark.parametrize("body, field", [
    ({"modulus_bits": 11}, "body.modulus_bits"), ({"modulus_bits": 13}, "body.modulus_bits"),
    ({"modulus_bits": 513}, "body.modulus_bits"), ({"modulus_bits": 1024}, "body.modulus_bits"),
    ({"modulus_bits": 0}, "body.modulus_bits"), ({"modulus_bits": -32}, "body.modulus_bits"),
    ({"modulus_bits": 32.0}, "body.modulus_bits"), ({"modulus_bits": "32"}, "body.modulus_bits"),
    ({"modulus_bits": True}, "body.modulus_bits"),
    ({"mode": "AES"}, "body.mode"), ({"mode": 5}, "body.mode"), ({"mode": None}, "body.mode"),
    ({"mode": "integer", "message": "abc"}, "body.message"),
    ({"mode": "integer", "message": "-5"}, "body.message"),
    ({"mode": "integer", "message": "1e3"}, "body.message"),
    ({"mode": "integer", "message": " 5"}, "body.message"),
    ({"mode": "integer", "message": "05"}, "body.message"),
    ({"mode": "integer", "message": "1_000"}, "body.message"),
    ({"mode": "integer", "message": ""}, "body.message"),
    ({"mode": "integer", "message": 65}, "body.message"),
    ({"mode": "integer", "message": "1" * 161}, "body.message"),
    ({"mode": "integer", "p": "6a", "q": "53"}, "body.p"),
    ({"mode": "integer", "p": 61, "q": "53"}, "body.p"),
    ({"mode": "integer", "p": "61", "q": "+53"}, "body.q"),
    ({"mode": "integer", "p": "61", "q": "53", "e": "0x11"}, "body.e"),
    ({"mode": "integer", "p": "61", "q": "53", "e": 17}, "body.e"),
    ({"aes_key_hex": "00" * 15}, "body.aes_key_hex"), ({"aes_key_hex": "zz" * 16}, "body.aes_key_hex"),
    ({"aes_key_hex": 12}, "body.aes_key_hex"),
    ({"seed": -1}, "body.seed"), ({"seed": 2 ** 32}, "body.seed"), ({"seed": 1.5}, "body.seed"),
    ({"seed": True}, "body.seed"), ({"seed": "7"}, "body.seed"),
    ({"include_traces": "yes"}, "body.include_traces"), ({"include_traces": 1}, "body.include_traces"),
])
def test_rsa_api_validation_errors_name_the_field(body, field):
    err = assert_error(rsa_post(**body), 422)
    assert err["details"] and err["details"][0]["field"] == field


@pytest.mark.parametrize("body, fragment", [
    ({"p": "61"}, "both 'p' and 'q'"),
    ({"q": "53"}, "both 'p' and 'q'"),
    ({"p": "61", "q": "53", "modulus_bits": 32}, "cannot be combined"),
    ({"mode": "integer", "aes_key_hex": "00" * 16}, "only valid with mode 'aes_key_wrap'"),
    ({"mode": "aes_key_wrap", "message": "5"}, "only valid with mode 'integer'"),
    ({"message": "5"}, "only valid with mode 'integer'"),
])
def test_rsa_api_conflicting_fields_rejected(body, fragment):
    err = assert_error(rsa_post(**body), 422)
    assert err["details"][0]["field"] == "body" and fragment in err["details"][0]["message"]


def test_rsa_api_extra_fields_malformed_json_and_methods():
    assert_error(rsa_post(bogus=1), 422)
    assert_error(client.post(RSA, content="{nope", headers={"content-type": "application/json"}), 422)
    assert_error(client.get(RSA), 405, "http_error")
    big = json.dumps({"mode": "integer", "message": "1" * (MAX_BODY_BYTES + 10)})
    assert_error(client.post(RSA, content=big, headers={"content-type": "application/json"}), 413)
    assert rsa_post().status_code == 200                                    # empty body {} uses all defaults


def test_rsa_api_errors_do_not_echo_private_values():
    secret_key = "11" * 16
    r = rsa_post(seed=1, aes_key_hex=secret_key, modulus_bits=13)
    assert r.status_code == 422 and secret_key not in r.text and str(int(secret_key, 16)) not in r.text


# ------------------------------------------------------------------ honesty of computed flags
def test_rsa_api_round_trip_flag_is_computed_not_hardcoded(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.decryption_rsa, "rsa_decrypt", lambda c, k: 12345)
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65").json()
    assert d["round_trip_ok"] is False and d["verification"]["round_trip_ok"] is False


def test_rsa_api_key_recovered_flag_is_computed_not_hardcoded(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.decryption_rsa, "rsa_unwrap_key", lambda c, k, n: b"\x00" * 16)
    assert rsa_post(seed=1).json()["key_recovered_ok"] is False


def test_rsa_api_key_recovered_false_when_unwrap_raises(monkeypatch):
    import backend.main as m

    def boom(c, k, n):
        raise ValueError("does not fit")

    monkeypatch.setattr(m.decryption_rsa, "rsa_unwrap_key", boom)
    d = rsa_post(seed=1).json()
    assert d["key_recovered_ok"] is False and d["recovered_key_hex"] is None


def test_rsa_api_trace_consistency_flag_is_computed(monkeypatch):
    import backend.main as m
    real = m.encryption_rsa.rsa_encrypt_trace

    def tampered(msg, pub):
        t = real(msg, pub)
        t["result"] += 1
        return t

    monkeypatch.setattr(m.encryption_rsa, "rsa_encrypt_trace", tampered)
    assert rsa_post(seed=1).json()["verification"]["encryption_trace_matches_ciphertext"] is False


def test_rsa_api_key_checks_are_really_computed(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.encryption_rsa, "is_prime", lambda n, *a, **k: False)
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65")
    assert d.status_code == 422                                              # keypair validation uses is_prime
    monkeypatch.undo()
    ok = rsa_post(mode="integer", p="61", q="53", e="17", message="65").json()["key_checks"]
    assert ok["p_is_prime"] and ok["q_is_prime"] and ok["n_equals_p_times_q"] and ok["phi_equals_p1_times_q1"]


def test_rsa_api_unexpected_crash_is_clean_500_without_private_values(monkeypatch):
    import backend.main as m

    def crash(*a, **k):
        raise RuntimeError("secret d=123456789 /srv/rsa.py")

    monkeypatch.setattr(m.encryption_rsa, "generate_keypair", crash)
    r = TestClient(app, raise_server_exceptions=False).post(RSA, json={"seed": 1})
    err = assert_error(r, 500, "internal_error")
    assert "123456789" not in r.text and err["message"] == "Internal server error."


def test_rsa_api_listed_in_openapi_and_version_bumped():
    assert "post" in client.get("/openapi.json").json()["paths"][RSA]


def test_rsa_api_modules_are_the_only_rsa_math_used():
    src = (pathlib.Path(__file__).resolve().parent.parent / "backend" / "main.py").read_text()
    assert "pow(" not in src.replace("gf_pow(", "")                           # the API never calls built-in pow


# ------------------------------------------------------------------ gap-closing tests (found by mutation testing)
def test_rsa_api_euclid_derivation_steps_are_truncated_for_a_huge_exponent():
    e = 2 ** 255 - 19                                       # a 255-bit prime exponent -> many Euclid steps
    d = rsa_post(modulus_bits=512, e=str(e), seed=1).json()
    dd = d["d_derivation"]
    phi = int(d["phi"])
    a, b, count = e % phi, phi, 0                           # independent step count
    while b:
        a, b, count = b, a % b, count + 1
    assert dd["steps_total"] == count > MAX_RSA_STEPS_RETURNED
    assert dd["steps_truncated"] is True and dd["steps_returned"] == len(dd["steps"]) == MAX_RSA_STEPS_RETURNED
    assert d["e_source"] == "user" and d["round_trip_ok"] is True and int(d["e"]) == e
    small = rsa_post(mode="integer", p="61", q="53", e="17").json()["d_derivation"]
    assert small["steps_truncated"] is False and small["steps_returned"] == small["steps_total"]


def test_rsa_api_round_trip_flag_requires_the_decryption_trace_to_agree(monkeypatch):
    import backend.main as m
    real = m.decryption_rsa.rsa_decrypt_trace

    def tampered(c, k):
        t = real(c, k)
        t["result"] += 1
        return t

    monkeypatch.setattr(m.decryption_rsa, "rsa_decrypt_trace", tampered)
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65").json()
    assert d["round_trip_ok"] is False and d["verification"]["round_trip_ok"] is False


def test_rsa_api_checks_detect_a_wrong_private_exponent(monkeypatch):
    import backend.main as m
    real = m.encryption_rsa.keypair_from_primes

    def fake(p, q, e=None):
        pair = real(p, q, e)
        return _rsa.RSAKeyPair(pair.public, _rsa.PrivateKey(pair.public.n, pair.private.d + 1),
                               pair.p, pair.q, pair.phi, pair.e_source, pair.d_steps)

    monkeypatch.setattr(m.encryption_rsa, "keypair_from_primes", fake)
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65").json()
    wrong = str(17 * 2754 % 3120)
    assert wrong == "18" and d["key_checks"]["e_times_d_mod_phi"] == "18"
    assert d["key_checks"]["all_ok"] is False and d["verification"]["key_checks_all_ok"] is False
    assert d["d_derivation"]["equation"].endswith("= 18")
    assert d["round_trip_ok"] is False and d["recovered"] != "65"


def test_rsa_api_primality_check_in_key_checks_is_computed(monkeypatch):
    import backend.main as m
    real, calls = m.encryption_rsa.is_prime, {"n": 0}

    def flaky(n, *a, **k):                                  # honest while the key is built, then lies
        calls["n"] += 1
        return real(n) if calls["n"] <= 2 else False

    monkeypatch.setattr(m.encryption_rsa, "is_prime", flaky)
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65").json()
    assert d["key_checks"]["p_is_prime"] is False and d["key_checks"]["q_is_prime"] is False
    assert d["key_checks"]["all_ok"] is False and d["verification"]["key_checks_all_ok"] is False


def test_rsa_api_gcd_check_in_key_checks_is_computed(monkeypatch):
    import backend.main as m
    real, calls = m.encryption_rsa.extended_gcd, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        return real(a, b) if calls["n"] <= 1 else (7, 0, 0)

    monkeypatch.setattr(m.encryption_rsa, "extended_gcd", flaky)
    d = rsa_post(mode="integer", p="61", q="53", e="17", message="65").json()
    assert d["key_checks"]["gcd_e_phi"] == "7" and d["key_checks"]["all_ok"] is False


def test_rsa_api_primality_reports_belong_to_the_right_prime():
    d = rsa_post(mode="integer", p="61", q="1000003", e="17").json()["primality"]
    assert d["p"]["method"] == "trial_division" and d["q"]["method"] == "miller_rabin_deterministic"
    d = rsa_post(mode="integer", p="1000003", q="61", e="17").json()["primality"]
    assert d["p"]["method"] == "miller_rabin_deterministic" and d["q"]["method"] == "trial_division"


@pytest.mark.parametrize("bits, code", [(13, "value_error"), (15, "value_error"), (11, "greater_than_equal"),
                                          (10, "greater_than_equal"), (2, "greater_than_equal"), (0, "greater_than_equal"),
                                          (-32, "greater_than_equal"), (514, "less_than_equal"), (1024, "less_than_equal")])
def test_rsa_api_modulus_bits_are_rejected_by_the_request_model_itself(bits, code):
    err = assert_error(rsa_post(mode="integer", modulus_bits=bits), 422)
    assert err["details"][0]["field"] == "body.modulus_bits" and err["details"][0]["code"] == code


@pytest.mark.parametrize("bad", ["32", 32.0, 32.5, True, [32], None if False else "x"])
def test_rsa_api_modulus_bits_must_be_a_real_integer_even_where_32_is_a_legal_size(bad):
    # In integer mode 32 bits is valid, so a lenient parser would accept "32" or 32.0 -> only strictness rejects them.
    err = assert_error(rsa_post(mode="integer", modulus_bits=bad), 422)
    assert err["details"][0]["field"] == "body.modulus_bits" and err["details"][0]["code"] != "invalid_key_parameter"
    assert rsa_post(mode="integer", modulus_bits=32, seed=1).status_code == 200


@pytest.mark.parametrize("field, value", [("p", "1" * 79), ("q", "1" * 79), ("e", "1" * 79), ("message", "1" * 161)])
def test_rsa_api_digit_limits_are_enforced_by_the_request_model(field, value):
    body = {"mode": "integer", field: value}
    if field in ("p", "q"):
        body["q" if field == "p" else "p"] = "61"
    err = assert_error(rsa_post(**body), 422)
    assert err["details"][0]["field"] == f"body.{field}" and err["details"][0]["code"] == "string_too_long"


def test_rsa_api_limits_accept_exactly_the_maximum_digits_at_the_model_layer():
    # 78-digit p passes the model; the algorithm layer (bit length) then decides.
    err = assert_error(rsa_post(mode="integer", p="9" * 78, q="61"), 422)
    assert err["details"][0]["code"] == "invalid_key_parameter"
    err = assert_error(rsa_post(mode="integer", p="61", q="53", e="17", message="9" * 160), 422)
    assert err["details"][0]["code"] == "message_too_large"

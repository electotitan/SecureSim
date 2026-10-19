"""API tests, appended checkpoint by checkpoint.

Checkpoint 3: POST /api/encode/huffman, POST /api/encode/binary, error handling.
"""
import json
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
    assert client.get("/api/health").json()["version"] == "0.4.0"


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

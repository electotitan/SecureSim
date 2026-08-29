"""
tests/test_pipeline.py

Phase 1 tests: prove the ARCHITECTURE works, independent of any real math
(which lands in Phase 2). Run with: pytest
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pipeline import Trace, IdentityStage, Pipeline, build_placeholder_pipeline


def test_identity_stage_passes_data_through():
    stage = IdentityStage("Test Stage")
    trace = Trace(stage_name=stage.name, input_data="abc")
    output = stage.run("abc", trace)
    assert output == "abc"
    assert len(trace.steps) == 1
    assert trace.steps[0]["action"] == "passthrough"


def test_pipeline_runs_all_stages_in_order():
    pipeline = build_placeholder_pipeline()
    result = pipeline.run("test message")
    assert len(result["traces"]) == 11
    expected_order = [
        "Source / Binary Encoding",
        "Huffman Compression",
        "RSA Encryption",
        "Packetization",
        "CRC Error Detection",
        "Graph Routing (Dijkstra)",
        "Channel Transmission (Noise)",
        "Hamming Error Correction",
        "RSA Decryption",
        "Huffman Decompression",
        "Binary / Source Decoding",
    ]
    actual_order = [t["stage_name"] for t in result["traces"]]
    assert actual_order == expected_order


def test_placeholder_pipeline_recovers_message_exactly():
    # With all IdentityStages, output must exactly equal input --
    # this is the baseline "clean channel" sanity check.
    pipeline = build_placeholder_pipeline()
    result = pipeline.run("Hello, secure world!")
    assert result["final_output"] == "Hello, secure world!"
    assert result["recovered_correctly"] is True


def test_trace_to_dict_has_expected_keys():
    trace = Trace(stage_name="X", input_data=1).finish(output_data=2, note="ok")
    d = trace.to_dict()
    assert set(d.keys()) == {"stage_name", "input_data", "output_data", "steps", "metadata"}
    assert d["metadata"]["note"] == "ok"


def test_custom_pipeline_can_be_assembled_from_stages():
    # Proves a team member can build a smaller pipeline for unit-testing
    # their own stage in isolation, before full integration.
    pipeline = Pipeline([IdentityStage("Stage A"), IdentityStage("Stage B")])
    result = pipeline.run("x")
    assert len(result["traces"]) == 2
    assert result["final_output"] == "x"

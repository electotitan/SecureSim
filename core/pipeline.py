"""
core/pipeline.py

This module defines the ARCHITECTURE for the whole project:
  - Trace: a record of what happened inside one stage (for the visualizer)
  - Stage: the interface every mathematical module (Huffman, RSA, Hamming, ...)
           must implement in Phase 2
  - Pipeline: chains stages together and produces a full run trace

Nothing in this file does real cryptography, compression, or coding theory yet.
Each Stage subclass here is an IDENTITY placeholder (data passes through
unchanged) so the full pipeline can be run and tested end-to-end starting
today. In Phase 2, each placeholder gets replaced with the real
first-principles implementation, but the shape (inputs, outputs, trace
format) will not change -- which is the point of nailing this down now.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trace:
    """
    Records everything a visualizer would need to animate one stage.

    stage_name : human-readable name, e.g. "Huffman Compression"
    input_data : what went into the stage (for display)
    output_data: what came out of the stage (for display)
    steps      : ordered list of intermediate steps, each a small dict.
                 Every stage defines its own step shape, e.g.:
                   Huffman: {"action": "merge_nodes", "left": ..., "right": ...}
                   Hamming: {"action": "compute_syndrome", "syndrome": [0,1,1]}
                 The visualizer will pattern-match on "action".
    metadata   : any extra scalar info worth surfacing (compression ratio,
                 chosen path length, number of bits flipped, etc.)
    """

    stage_name: str
    input_data: Any
    output_data: Any = None
    steps: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def log_step(self, **kwargs) -> None:
        """Convenience method for a stage to record one intermediate step."""
        self.steps.append(kwargs)

    def finish(self, output_data: Any, **metadata) -> "Trace":
        self.output_data = output_data
        self.metadata.update(metadata)
        return self

    def to_dict(self) -> dict:
        return {
            "stage_name": self.stage_name,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "steps": self.steps,
            "metadata": self.metadata,
        }


class Stage(ABC):
    """
    Every mathematical module in the pipeline (encoding, compression,
    encryption, packetization, error detection, routing, channel noise,
    error correction, decryption, decompression, decoding) implements this
    interface.

    A Stage takes some data in, produces some data out, and records a Trace
    of what happened along the way so it can be visualized.
    """

    name: str = "UnnamedStage"

    @abstractmethod
    def run(self, data: Any, trace: Trace) -> Any:
        """
        Perform this stage's transformation on `data`.
        Must call trace.log_step(...) for each meaningful intermediate
        operation, and return the output data.
        Do NOT call trace.finish() here -- the Pipeline does that.
        """
        raise NotImplementedError


class IdentityStage(Stage):
    """
    Placeholder stage used until Phase 2 implements the real math.
    Passes data through unchanged but still participates fully in the
    trace/logging contract, so downstream code (pipeline runner, API,
    frontend, tests) can be built and tested against it right now.
    """

    def __init__(self, name: str):
        self.name = name

    def run(self, data: Any, trace: Trace) -> Any:
        trace.log_step(action="passthrough", note=f"{self.name} not yet implemented (Phase 2)")
        return data


class Pipeline:
    """
    Chains a sequence of Stages together and runs data through all of them,
    collecting one Trace per stage plus overall pipeline metadata.
    """

    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, message: str) -> dict:
        """
        Run `message` through every stage in order.
        Returns a dict with the final output and the full list of traces,
        i.e. exactly what the API layer will hand to the frontend.
        """
        data: Any = message
        traces: list[Trace] = []

        for stage in self.stages:
            trace = Trace(stage_name=stage.name, input_data=data)
            output = stage.run(data, trace)
            trace.finish(output_data=output)
            traces.append(trace)
            data = output

        return {
            "original_message": message,
            "final_output": data,
            "recovered_correctly": data == message,
            "traces": [t.to_dict() for t in traces],
        }


def build_default_pipeline(
    channel_flip_probability: float = 0.0,
    key_bits: int = 16,
    keypair: dict | None = None,
) -> Pipeline:
    """
    Wires up the full pipeline using the real Phase 2 implementations.

    NOTE on stage order vs. the original Phase-0/Phase-1 placeholder list:
    Hamming error correction only makes sense if it ENCODES before the
    noisy channel (adding redundancy) and CORRECTS after it (using that
    redundancy). So "Hamming Error Correction" from the abstract is
    implemented here as two stages that sandwich routing + the channel,
    rather than one stage placed after it. Likewise CRC is split into an
    attach/check pair for the same reason. This is a refinement of the
    architecture, not a change to its goals -- every stage named in the
    abstract is still here, just correctly positioned relative to where
    corruption is actually introduced.

    A single RSA keypair is generated once (or supplied) and shared
    between the encryption and decryption stages, since in a real system
    the private key never travels through the pipeline itself.
    """
    from core.encoding import BinaryEncodingStage, BinaryDecodingStage
    from core.huffman import HuffmanCompressionStage, HuffmanDecompressionStage
    from core.rsa import RSAEncryptionStage, RSADecryptionStage, generate_keypair
    from core.packetization import PacketizationStage, ReassemblyStage
    from core.hamming import HammingEncodeStage, HammingCorrectStage
    from core.crc import CRCAttachStage, CRCCheckStage
    from core.graph_routing import GraphRoutingStage
    from core.channel import ChannelStage

    keypair = keypair or generate_keypair(key_bits)

    stages = [
        BinaryEncodingStage(),
        HuffmanCompressionStage(),
        RSAEncryptionStage(keypair),
        PacketizationStage(),
        HammingEncodeStage(),
        CRCAttachStage(),
        GraphRoutingStage(),
        ChannelStage(flip_probability=channel_flip_probability),
        CRCCheckStage(),
        HammingCorrectStage(),
        ReassemblyStage(),
        RSADecryptionStage(keypair),
        HuffmanDecompressionStage(),
        BinaryDecodingStage(),
    ]
    return Pipeline(stages)


def build_placeholder_pipeline() -> Pipeline:
    """
    The original Phase 1 all-IdentityStage pipeline, kept around for the
    architecture-only tests (proves the Stage/Trace/Pipeline contract in
    isolation from any real math).
    """
    stage_names = [
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
    return Pipeline([IdentityStage(name) for name in stage_names])


if __name__ == "__main__":
    # Quick manual smoke test: run the real pipeline end-to-end, clean channel.
    pipeline = build_default_pipeline(channel_flip_probability=0.0)
    result = pipeline.run("Hello, secure world!")
    print(f"Original:  {result['original_message']}")
    print(f"Final:     {result['final_output']}")
    print(f"Recovered correctly: {result['recovered_correctly']}")
    print(f"Number of stages run: {len(result['traces'])}")

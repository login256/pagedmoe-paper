#!/usr/bin/env python3
"""Measure one Qwen3.5-397B routed expert after llama.cpp Qx_K quantization.

The script intentionally loads only one routed expert. Each matrix is
quantized, dequantized, and multiplied immediately so that multiple quantized
copies are never resident at the same time.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path("/mnt/data/Qwen3.5-397B-A17B")
DEFAULT_LLAMA_CPP_LIB = Path.home() / "llama.cpp/build/bin/libggml-base.so.0"

HIDDEN_SIZE = 4096
MOE_INTERMEDIATE_SIZE = 1024
QK_K = 256


@dataclass(frozen=True)
class QuantType:
    nominal_bits: int
    name: str
    block_bytes: int
    note: str = ""

    @property
    def effective_bpw(self) -> float:
        return self.block_bytes * 8 / QK_K


QUANT_TYPES = {
    quant.name: quant
    for quant in (
        QuantType(2, "Q2_K", 84),
        QuantType(3, "Q3_K", 110),
        QuantType(4, "Q4_K", 144),
        QuantType(5, "Q5_K", 176),
        QuantType(6, "Q6_K", 210),
        QuantType(
            8,
            "Q8_K",
            292,
            "llama.cpp internal intermediate/dot-product format",
        ),
    )
}
UNAVAILABLE = {
    1: "llama.cpp has no Q1_K format",
    7: "llama.cpp has no Q7_K format",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep llama.cpp Qx_K formats for one Qwen3.5-397B routed expert."
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--llama-cpp-lib", type=Path, default=DEFAULT_LLAMA_CPP_LIB)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--quant-types",
        nargs="+",
        default=list(QUANT_TYPES),
        choices=list(QUANT_TYPES),
    )
    return parser.parse_args()


class LlamaCppQK:
    """Thin ctypes binding for llama.cpp's reference Qx_K row functions."""

    def __init__(self, library_path: Path):
        self.library_path = library_path.resolve()
        self.lib = ctypes.CDLL(str(self.library_path))
        self.functions = {}
        for quant in QUANT_TYPES.values():
            quantize = getattr(self.lib, f"quantize_row_q{quant.nominal_bits}_K_ref")
            dequantize = getattr(self.lib, f"dequantize_row_q{quant.nominal_bits}_K")
            quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            quantize.restype = None
            dequantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            dequantize.restype = None
            self.functions[quant.name] = (quantize, dequantize)

    def quantize_dequantize(
        self, source: np.ndarray, quant: QuantType
    ) -> tuple[np.ndarray, int]:
        source = np.ascontiguousarray(source, dtype=np.float32)
        if source.size % QK_K:
            raise ValueError(f"{source.size} elements are not divisible by QK_K={QK_K}")

        quantized_bytes = source.size // QK_K * quant.block_bytes
        packed = np.empty(quantized_bytes, dtype=np.uint8)
        dequantized = np.empty_like(source)
        quantize, dequantize = self.functions[quant.name]
        quantize(source.ctypes.data, packed.ctypes.data, source.size)
        dequantize(packed.ctypes.data, dequantized.ctypes.data, source.size)
        return dequantized, quantized_bytes


def tensor_paths(model_root: Path, layer_index: int) -> tuple[str, Path, str, Path]:
    with (model_root / "model.safetensors.index.json").open("rb") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    gate_up_key = f"model.language_model.layers.{layer_index}.mlp.experts.gate_up_proj"
    down_key = f"model.language_model.layers.{layer_index}.mlp.experts.down_proj"
    return (
        gate_up_key,
        model_root / weight_map[gate_up_key],
        down_key,
        model_root / weight_map[down_key],
    )


def load_one_expert(
    model_root: Path, layer_index: int, expert_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str]]:
    gate_up_key, gate_up_path, down_key, down_path = tensor_paths(model_root, layer_index)

    with safe_open(gate_up_path, framework="pt", device="cpu") as weights:
        gate_up = weights.get_slice(gate_up_key)[expert_id].contiguous()
    with safe_open(down_path, framework="pt", device="cpu") as weights:
        down = weights.get_slice(down_key)[expert_id].contiguous()

    expected_gate_up = (MOE_INTERMEDIATE_SIZE * 2, HIDDEN_SIZE)
    expected_down = (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
    if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 weights, got {gate_up.dtype=} and {down.dtype=}")
    if tuple(gate_up.shape) != expected_gate_up:
        raise ValueError(f"unexpected gate_up shape {tuple(gate_up.shape)}")
    if tuple(down.shape) != expected_down:
        raise ValueError(f"unexpected down shape {tuple(down.shape)}")

    gate = gate_up[:MOE_INTERMEDIATE_SIZE].contiguous()
    up = gate_up[MOE_INTERMEDIATE_SIZE:].contiguous()
    del gate_up
    return gate, up, down, {
        "gate_up_tensor": gate_up_key,
        "gate_up_shard": str(gate_up_path),
        "down_tensor": down_key,
        "down_shard": str(down_path),
    }


def unquantized_expert(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    hidden_state_bf16: torch.Tensor,
) -> torch.Tensor:
    hidden_state = hidden_state_bf16.float()
    gate_out = torch.mv(gate.float(), hidden_state)
    up_out = torch.mv(up.float(), hidden_state)
    intermediate = (F.silu(gate_out) * up_out).to(torch.bfloat16).float()
    return torch.mv(down.float(), intermediate)


def quantized_mv(
    qk: LlamaCppQK,
    quant: QuantType,
    weight_bf16: torch.Tensor,
    vector: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    weight_fp32 = weight_bf16.float().numpy()
    dequantized, quantized_bytes = qk.quantize_dequantize(weight_fp32, quant)
    output = torch.mv(torch.from_numpy(dequantized), vector)
    return output, quantized_bytes


def quantized_expert(
    qk: LlamaCppQK,
    quant: QuantType,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    hidden_state_bf16: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    hidden_state = hidden_state_bf16.float()
    gate_out, gate_bytes = quantized_mv(qk, quant, gate, hidden_state)
    up_out, up_bytes = quantized_mv(qk, quant, up, hidden_state)
    intermediate = (F.silu(gate_out) * up_out).to(torch.bfloat16).float()
    output, down_bytes = quantized_mv(qk, quant, down, intermediate)
    return output, gate_bytes + up_bytes + down_bytes


def metrics(reference: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    diff = output.float() - reference.float()
    reference_norm = torch.linalg.vector_norm(reference.float()).item()
    return {
        "relative_l2_error": torch.linalg.vector_norm(diff).item()
        / max(reference_norm, 1e-12),
        "l2_error": torch.linalg.vector_norm(diff).item(),
        "max_absolute_error": diff.abs().max().item(),
        "mean_absolute_error": diff.abs().mean().item(),
        "cosine_similarity": F.cosine_similarity(
            reference.float(), output.float(), dim=0
        ).item(),
        "reference_l2_norm": reference_norm,
        "output_l2_norm": torch.linalg.vector_norm(output.float()).item(),
    }


def csv_fieldnames() -> list[str]:
    return [
        "nominal_bits",
        "quant_type",
        "status",
        "effective_bits_per_weight",
        "quantized_expert_bytes",
        "relative_l2_error",
        "l2_error",
        "max_absolute_error",
        "mean_absolute_error",
        "cosine_similarity",
        "reference_l2_norm",
        "output_l2_norm",
        "elapsed_seconds",
        "note",
    ]


def unavailable_row(nominal_bits: int) -> dict[str, object]:
    return {
        "nominal_bits": nominal_bits,
        "quant_type": f"Q{nominal_bits}_K",
        "status": "unavailable",
        "note": UNAVAILABLE[nominal_bits],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames(), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    selected_quant_types = [QUANT_TYPES[name] for name in args.quant_types]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden_state_bf16 = torch.randn(
        HIDDEN_SIZE, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)

    started = time.time()
    gate, up, down, tensor_metadata = load_one_expert(
        args.model_root, args.layer_index, args.expert_id
    )
    reference = unquantized_expert(gate, up, down, hidden_state_bf16)
    qk = LlamaCppQK(args.llama_cpp_lib)

    rows_by_bit: dict[int, dict[str, object]] = {
        bit: unavailable_row(bit) for bit in UNAVAILABLE
    }
    print(
        f"model={args.model_root} layer={args.layer_index} expert={args.expert_id} "
        f"seed={args.seed}"
    )
    print("quant  effective_bpw  relative_l2_error  cosine_similarity  packed_mib")
    for quant in selected_quant_types:
        quant_started = time.time()
        output, quantized_bytes = quantized_expert(
            qk, quant, gate, up, down, hidden_state_bf16
        )
        row: dict[str, object] = {
            "nominal_bits": quant.nominal_bits,
            "quant_type": quant.name,
            "status": "available",
            "effective_bits_per_weight": quant.effective_bpw,
            "quantized_expert_bytes": quantized_bytes,
            "elapsed_seconds": time.time() - quant_started,
            "note": quant.note,
        }
        row.update(metrics(reference, output))
        rows_by_bit[quant.nominal_bits] = row
        print(
            f"{quant.name:>5}  {quant.effective_bpw:>13.4f}  "
            f"{row['relative_l2_error']:>17.8e}  "
            f"{row['cosine_similarity']:>17.8f}  "
            f"{quantized_bytes / 1024**2:>10.3f}"
        )
        del output
        gc.collect()

    rows = [rows_by_bit[bit] for bit in sorted(rows_by_bit)]
    csv_path = args.output_dir / "qwen35_expert_qk_relative_error.csv"
    metadata_path = args.output_dir / "qwen35_expert_qk_relative_error.json"
    write_csv(csv_path, rows)

    metadata = {
        "model_root": str(args.model_root.resolve()),
        "llama_cpp_library": str(qk.library_path),
        "layer_index": args.layer_index,
        "expert_id": args.expert_id,
        "hidden_state": {
            "description": "one deterministic synthetic BF16 hidden state sampled from N(0, 1)",
            "seed": args.seed,
            "size": HIDDEN_SIZE,
        },
        "expert": {
            "architecture": "SiLU(gate_proj(x)) * up_proj(x), followed by down_proj",
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "parameter_count": sum(tensor.numel() for tensor in (gate, up, down)),
            "original_weight_bytes": sum(
                tensor.numel() * tensor.element_size() for tensor in (gate, up, down)
            ),
        },
        "tensor_sources": tensor_metadata,
        "quantization": {
            "description": (
                "llama.cpp quantize_row_qx_K_ref followed by dequantize_row_qx_K; "
                "each matrix is processed and multiplied before the next matrix"
            ),
            "qk_block_elements": QK_K,
            "selected_types": [asdict(quant) for quant in selected_quant_types],
            "unavailable_types": UNAVAILABLE,
        },
        "reference_output_l2_norm": torch.linalg.vector_norm(reference.float()).item(),
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "csv": str(csv_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"csv={csv_path}")
    print(f"metadata={metadata_path}")
    print(f"max_rss_mib={metadata['max_rss_mib']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

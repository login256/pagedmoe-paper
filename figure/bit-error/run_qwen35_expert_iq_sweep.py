#!/usr/bin/env python3
"""Measure one Qwen3.5-397B routed expert after llama.cpp IQ quantization.

llama.cpp defines IQ primitive formats only from IQ1 through IQ4. This runner
records every directly representable IQ primitive and marks IQ5 through IQ8 as
unavailable. It loads only one routed expert and processes one matrix at a
time. Quantization is parallelized across independent matrix rows when
--workers is greater than one.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import math
import os
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

from run_qwen35_expert_qk_sweep import (
    DEFAULT_LLAMA_CPP_LIB,
    DEFAULT_MODEL_ROOT,
    HIDDEN_SIZE,
    load_one_expert,
    metrics,
    unquantized_expert,
)


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class IQType:
    nominal_bits: int
    name: str
    ggml_type: int
    dequantize_suffix: str


IQ_TYPES = {
    quant.name: quant
    for quant in (
        IQType(1, "IQ1_S", 19, "iq1_s"),
        IQType(1, "IQ1_M", 29, "iq1_m"),
        IQType(2, "IQ2_XXS", 16, "iq2_xxs"),
        IQType(2, "IQ2_XS", 17, "iq2_xs"),
        IQType(2, "IQ2_S", 22, "iq2_s"),
        IQType(3, "IQ3_XXS", 18, "iq3_xxs"),
        IQType(3, "IQ3_S", 21, "iq3_s"),
        IQType(4, "IQ4_XS", 23, "iq4_xs"),
        IQType(4, "IQ4_NL", 20, "iq4_nl"),
    )
}
UNAVAILABLE = {
    bit: f"llama.cpp has no IQ{bit} primitive format" for bit in range(5, 9)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep llama.cpp IQ primitives for one Qwen3.5-397B routed expert."
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--llama-cpp-lib", type=Path, default=DEFAULT_LLAMA_CPP_LIB)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of independent matrix-row quantization workers",
    )
    parser.add_argument(
        "--tasks-per-worker",
        type=int,
        default=4,
        help="split each matrix into approximately workers * this many tasks",
    )
    parser.add_argument(
        "--iq-types",
        nargs="+",
        default=list(IQ_TYPES),
        choices=list(IQ_TYPES),
    )
    return parser.parse_args()


class LlamaCppIQ:
    """Thin ctypes binding for llama.cpp IQ quantization primitives."""

    def __init__(self, library_path: Path):
        self.library_path = library_path.resolve()
        self.lib = ctypes.CDLL(str(self.library_path))
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_init.restype = None
        self.lib.ggml_quantize_free.argtypes = []
        self.lib.ggml_quantize_free.restype = None
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t

        self.dequantize_functions = {}
        for quant in IQ_TYPES.values():
            dequantize = getattr(self.lib, f"dequantize_row_{quant.dequantize_suffix}")
            dequantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            dequantize.restype = None
            self.dequantize_functions[quant.name] = dequantize

    def row_size(self, quant: IQType, columns: int) -> int:
        return int(self.lib.ggml_row_size(quant.ggml_type, columns))

    def effective_bpw(self, quant: IQType, columns: int) -> float:
        return self.row_size(quant, columns) * 8 / columns

    def free_tables(self) -> None:
        self.lib.ggml_quantize_free()

    def quantize_dequantize(
        self,
        source: np.ndarray,
        quant: IQType,
        importance: np.ndarray,
        *,
        workers: int,
        tasks_per_worker: int,
    ) -> tuple[np.ndarray, int]:
        source = np.ascontiguousarray(source, dtype=np.float32)
        importance = np.ascontiguousarray(importance, dtype=np.float32)
        if source.ndim != 2:
            raise ValueError(f"expected a matrix, got shape {source.shape}")
        rows, columns = source.shape
        if importance.shape != (columns,):
            raise ValueError(
                f"expected importance shape {(columns,)}, got {importance.shape}"
            )

        row_size = self.row_size(quant, columns)
        packed = np.empty(rows * row_size, dtype=np.uint8)
        dequantized = np.empty_like(source)
        quantize = self.lib.ggml_quantize_chunk
        dequantize = self.dequantize_functions[quant.name]
        self.lib.ggml_quantize_init(quant.ggml_type)

        task_count = max(1, workers * tasks_per_worker)
        rows_per_task = max(1, math.ceil(rows / task_count))
        row_ranges = [
            (start, min(rows, start + rows_per_task))
            for start in range(0, rows, rows_per_task)
        ]

        source_ptr = source.ctypes.data
        packed_ptr = packed.ctypes.data
        dequantized_ptr = dequantized.ctypes.data
        importance_ptr = importance.ctypes.data

        def process_rows(row_range: tuple[int, int]) -> int:
            start_row, end_row = row_range
            row_count = end_row - start_row
            start_element = start_row * columns
            quantized_bytes = quantize(
                quant.ggml_type,
                source_ptr,
                packed_ptr,
                start_element,
                row_count,
                columns,
                importance_ptr,
            )
            dequantize(
                packed_ptr + start_row * row_size,
                dequantized_ptr + start_element * source.itemsize,
                row_count * columns,
            )
            return int(quantized_bytes)

        if workers == 1:
            quantized_bytes = sum(process_rows(row_range) for row_range in row_ranges)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                quantized_bytes = sum(executor.map(process_rows, row_ranges))
        return dequantized, quantized_bytes


def quantized_mv(
    iq: LlamaCppIQ,
    quant: IQType,
    weight_bf16: torch.Tensor,
    vector: torch.Tensor,
    importance: torch.Tensor,
    *,
    workers: int,
    tasks_per_worker: int,
) -> tuple[torch.Tensor, int]:
    weight_fp32 = weight_bf16.float().numpy()
    dequantized, quantized_bytes = iq.quantize_dequantize(
        weight_fp32,
        quant,
        importance.float().numpy(),
        workers=workers,
        tasks_per_worker=tasks_per_worker,
    )
    output = torch.mv(torch.from_numpy(dequantized), vector)
    return output, quantized_bytes


def quantized_expert(
    iq: LlamaCppIQ,
    quant: IQType,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    hidden_state_bf16: torch.Tensor,
    *,
    workers: int,
    tasks_per_worker: int,
) -> tuple[torch.Tensor, int]:
    hidden_state = hidden_state_bf16.float()
    hidden_importance = hidden_state.square().contiguous()
    gate_out, gate_bytes = quantized_mv(
        iq,
        quant,
        gate,
        hidden_state,
        hidden_importance,
        workers=workers,
        tasks_per_worker=tasks_per_worker,
    )
    up_out, up_bytes = quantized_mv(
        iq,
        quant,
        up,
        hidden_state,
        hidden_importance,
        workers=workers,
        tasks_per_worker=tasks_per_worker,
    )
    intermediate = (F.silu(gate_out) * up_out).to(torch.bfloat16).float()
    down_out, down_bytes = quantized_mv(
        iq,
        quant,
        down,
        intermediate,
        intermediate.square().contiguous(),
        workers=workers,
        tasks_per_worker=tasks_per_worker,
    )
    return down_out, gate_bytes + up_bytes + down_bytes


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
        "quant_type": f"IQ{nominal_bits}",
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
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.tasks_per_worker < 1:
        raise ValueError("--tasks-per-worker must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    selected_iq_types = [IQ_TYPES[name] for name in args.iq_types]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden_state_bf16 = torch.randn(
        HIDDEN_SIZE, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)

    started = time.time()
    gate, up, down, tensor_metadata = load_one_expert(
        args.model_root, args.layer_index, args.expert_id
    )
    reference = unquantized_expert(gate, up, down, hidden_state_bf16)
    iq = LlamaCppIQ(args.llama_cpp_lib)

    rows: list[dict[str, object]] = []
    print(
        f"model={args.model_root} layer={args.layer_index} expert={args.expert_id} "
        f"seed={args.seed} workers={args.workers}"
    )
    print("quant    effective_bpw  relative_l2_error  cosine_similarity  packed_mib  elapsed_s")
    try:
        for quant in selected_iq_types:
            quant_started = time.time()
            print(f"starting {quant.name}", flush=True)
            output, quantized_bytes = quantized_expert(
                iq,
                quant,
                gate,
                up,
                down,
                hidden_state_bf16,
                workers=args.workers,
                tasks_per_worker=args.tasks_per_worker,
            )
            row: dict[str, object] = {
                "nominal_bits": quant.nominal_bits,
                "quant_type": quant.name,
                "status": "available",
                "effective_bits_per_weight": iq.effective_bpw(quant, HIDDEN_SIZE),
                "quantized_expert_bytes": quantized_bytes,
                "elapsed_seconds": time.time() - quant_started,
                "note": "importance weights derived from the measured hidden state",
            }
            row.update(metrics(reference, output))
            rows.append(row)
            print(
                f"{quant.name:>7}  {row['effective_bits_per_weight']:>13.4f}  "
                f"{row['relative_l2_error']:>17.8e}  "
                f"{row['cosine_similarity']:>17.8f}  "
                f"{quantized_bytes / 1024**2:>10.3f}  "
                f"{row['elapsed_seconds']:>9.3f}",
                flush=True,
            )
            del output
            iq.free_tables()
            gc.collect()
    finally:
        iq.free_tables()

    rows.extend(unavailable_row(bit) for bit in UNAVAILABLE)
    csv_path = args.output_dir / "qwen35_expert_iq_relative_error.csv"
    metadata_path = args.output_dir / "qwen35_expert_iq_relative_error.json"
    write_csv(csv_path, rows)

    metadata = {
        "model_root": str(args.model_root.resolve()),
        "llama_cpp_library": str(iq.library_path),
        "layer_index": args.layer_index,
        "expert_id": args.expert_id,
        "hidden_state": {
            "description": "one deterministic synthetic BF16 hidden state sampled from N(0, 1)",
            "seed": args.seed,
            "size": HIDDEN_SIZE,
        },
        "importance_matrix": {
            "description": (
                "per-input weights derived from the measured activation: hidden_state^2 "
                "for gate/up and quantized-path intermediate^2 for down"
            ),
        },
        "expert": {
            "architecture": "SiLU(gate_proj(x)) * up_proj(x), followed by down_proj",
            "hidden_size": HIDDEN_SIZE,
            "parameter_count": sum(tensor.numel() for tensor in (gate, up, down)),
            "original_weight_bytes": sum(
                tensor.numel() * tensor.element_size() for tensor in (gate, up, down)
            ),
        },
        "tensor_sources": tensor_metadata,
        "quantization": {
            "description": (
                "llama.cpp ggml_quantize_chunk followed by the matching "
                "dequantize_row_iq* function; each matrix is processed and multiplied "
                "before the next matrix"
            ),
            "selected_types": [asdict(quant) for quant in selected_iq_types],
            "unavailable_types": UNAVAILABLE,
            "workers": args.workers,
            "tasks_per_worker": args.tasks_per_worker,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
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

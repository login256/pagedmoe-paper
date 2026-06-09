#!/usr/bin/env python3
"""Add Qwen3.5 GPTQ-Int4 weight error as a reference series.

This script appends/replaces ``method=gptq_int4`` rows in the existing
ktensor-lite direct-vs-readout CSV files. It does not rerun ktensor-lite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
from safetensors import safe_open


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SUBEXP_DIR = REPO_ROOT / "subexp"
if str(SUBEXP_DIR) not in sys.path:
    sys.path.insert(0, str(SUBEXP_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pagedmoe_ktensor_lite_io import load_qwen35_expert_weights  # noqa: E402
from run_qwen35_layer_ktensor_lite_weight_error import (  # noqa: E402
    DEFAULT_LAYER_INDEX,
    DEFAULT_MODEL_ROOT,
    DEFAULT_OUTPUT_STEM,
    Accumulator,
    aggregate_rows,
    csv_fieldnames,
    output_paths,
    read_existing_rows,
    write_summary,
)


DEFAULT_GPTQ_ROOT = Path("/mnt/data/Qwen3.5-397B-A17B-GPTQ-Int4")
METHOD = "gptq_int4"
READOUT_SOURCE = "gptq_int4"
TENSOR_TO_PROJ = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}
SHIFTS = torch.arange(0, 32, 4, dtype=torch.int64)
MASK = (1 << 4) - 1


@dataclass(frozen=True)
class WorkerConfig:
    model_root: str
    gptq_root: str
    layer_index: int
    torch_threads: int


@dataclass
class ExpertResult:
    expert_id: int
    expert_rows: list[dict[str, Any]] = field(default_factory=list)
    tensor_rows: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--gptq-root", type=Path, default=DEFAULT_GPTQ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--expert-offset", type=int, default=0)
    parser.add_argument("--expert-limit", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def load_weight_map(root: Path) -> dict[str, str]:
    with (root / "model.safetensors.index.json").open("rb") as index_file:
        return json.load(index_file)["weight_map"]


def load_gptq_tensor(root: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(root / weight_map[key], framework="pt", device="cpu") as weights:
        return weights.get_tensor(key)


def unpack_qweight(qweight: torch.Tensor, in_features: int) -> torch.Tensor:
    values = (qweight.to(torch.int64).unsqueeze(1) >> SHIFTS.view(1, -1, 1)) & MASK
    return values.reshape(qweight.shape[0] * 8, qweight.shape[1])[:in_features]


def unpack_qzeros(qzeros: torch.Tensor, out_features: int) -> torch.Tensor:
    values = (qzeros.to(torch.int64).unsqueeze(-1) >> SHIFTS) & MASK
    return values.reshape(qzeros.shape[0], -1)[:, :out_features]


def dequant_gptq_projection(
    root: Path,
    weight_map: dict[str, str],
    *,
    layer_index: int,
    expert_id: int,
    projection: str,
) -> torch.Tensor:
    prefix = f"model.language_model.layers.{layer_index}.mlp.experts.{expert_id}.{projection}"
    qweight = load_gptq_tensor(root, weight_map, f"{prefix}.qweight")
    qzeros = load_gptq_tensor(root, weight_map, f"{prefix}.qzeros")
    scales = load_gptq_tensor(root, weight_map, f"{prefix}.scales").float()
    g_idx = load_gptq_tensor(root, weight_map, f"{prefix}.g_idx").long()

    in_features = int(g_idx.numel())
    out_features = int(scales.shape[1])
    q = unpack_qweight(qweight, in_features).float()
    zeros = unpack_qzeros(qzeros, out_features).float()
    weight_in_out = (q - zeros[g_idx]) * scales[g_idx]
    return weight_in_out.t().contiguous()


def row_from_accumulator(
    *,
    layer_index: int,
    expert_id: int | str,
    tensor: str,
    accumulator: Accumulator,
    elapsed_seconds: float | str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "layer_index": layer_index,
        "expert_id": expert_id,
        "tensor": tensor,
        "bit": 4,
        "method": METHOD,
        "readout_source": READOUT_SOURCE,
        "sampling_rate": "",
        "elapsed_seconds": elapsed_seconds,
    }
    row.update(accumulator.metrics())
    return row


def process_expert(expert_id: int, config_dict: dict[str, Any]) -> ExpertResult:
    config = WorkerConfig(**config_dict)
    torch.set_num_threads(max(1, config.torch_threads))
    torch.set_grad_enabled(False)
    started = time.time()

    model_root = Path(config.model_root)
    gptq_root = Path(config.gptq_root)
    weight_map = load_weight_map(gptq_root)
    gate, up, down = load_qwen35_expert_weights(model_root, config.layer_index, expert_id)
    references = {
        "gate": gate.float().contiguous(),
        "up": up.float().contiguous(),
        "down": down.float().contiguous(),
    }

    expert_accum = Accumulator()
    tensor_rows: list[dict[str, Any]] = []
    for tensor_name, projection in TENSOR_TO_PROJ.items():
        reference = references[tensor_name]
        dequantized = dequant_gptq_projection(
            gptq_root,
            weight_map,
            layer_index=config.layer_index,
            expert_id=expert_id,
            projection=projection,
        )
        if tuple(dequantized.shape) != tuple(reference.shape):
            raise ValueError(
                f"{tensor_name} shape mismatch: GPTQ {tuple(dequantized.shape)} vs reference {tuple(reference.shape)}"
            )
        reference_flat = reference.view(-1)
        dequantized_flat = dequantized.view(-1)
        reference_l2_sq = float(torch.dot(reference_flat, reference_flat).item())
        tensor_accum = Accumulator()
        tensor_accum.add(reference_flat, dequantized_flat, reference_l2_sq)
        expert_accum.add(reference_flat, dequantized_flat, reference_l2_sq)
        tensor_rows.append(
            row_from_accumulator(
                layer_index=config.layer_index,
                expert_id=expert_id,
                tensor=tensor_name,
                accumulator=tensor_accum,
                elapsed_seconds="",
            )
        )

    elapsed = time.time() - started
    expert_rows = [
        row_from_accumulator(
            layer_index=config.layer_index,
            expert_id=expert_id,
            tensor="all",
            accumulator=expert_accum,
            elapsed_seconds=elapsed,
        )
    ]
    for row in tensor_rows:
        row["elapsed_seconds"] = elapsed
    return ExpertResult(
        expert_id=expert_id,
        expert_rows=expert_rows,
        tensor_rows=tensor_rows,
        elapsed_seconds=elapsed,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def remove_method(rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("method") != method]


def run(args: argparse.Namespace) -> int:
    if args.expert_offset < 0 or args.expert_limit < 1:
        raise ValueError("--expert-offset must be non-negative and --expert-limit positive")
    if not (args.gptq_root / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"GPTQ index not found under {args.gptq_root}")

    output_stem = args.output_stem
    if args.layer_index != DEFAULT_LAYER_INDEX and output_stem == DEFAULT_OUTPUT_STEM:
        output_stem = f"qwen35_layer{args.layer_index}_ktensor_lite_weight_error"
    paths = output_paths(args.output_dir, output_stem)
    if not paths["experts"].exists() or not paths["tensors"].exists():
        raise FileNotFoundError(
            "run the ktensor-lite experiment first so experts/tensors CSVs exist"
        )

    expert_end = min(512, args.expert_offset + args.expert_limit)
    expert_ids = list(range(args.expert_offset, expert_end))
    config = WorkerConfig(
        model_root=str(args.model_root),
        gptq_root=str(args.gptq_root),
        layer_index=args.layer_index,
        torch_threads=args.torch_threads,
    )
    config_dict = asdict(config)

    print(
        f"layer={args.layer_index} experts={args.expert_offset}..{expert_end - 1} "
        f"method={METHOD} workers={args.workers}",
        flush=True,
    )

    gptq_expert_rows: list[dict[str, Any]] = []
    gptq_tensor_rows: list[dict[str, Any]] = []
    started = time.time()
    if args.workers <= 1:
        for index, expert_id in enumerate(expert_ids, start=1):
            result = process_expert(expert_id, config_dict)
            gptq_expert_rows.extend(result.expert_rows)
            gptq_tensor_rows.extend(result.tensor_rows)
            print(
                f"expert={expert_id} elapsed={result.elapsed_seconds:.2f}s "
                f"progress={index}/{len(expert_ids)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_expert, expert_id, config_dict): expert_id
                for expert_id in expert_ids
            }
            for index, future in enumerate(as_completed(futures), start=1):
                expert_id = futures[future]
                result = future.result()
                gptq_expert_rows.extend(result.expert_rows)
                gptq_tensor_rows.extend(result.tensor_rows)
                print(
                    f"expert={expert_id} elapsed={result.elapsed_seconds:.2f}s "
                    f"progress={index}/{len(expert_ids)}",
                    flush=True,
                )

    existing_expert_rows = remove_method(read_existing_rows(paths["experts"]), METHOD)
    existing_tensor_rows = remove_method(read_existing_rows(paths["tensors"]), METHOD)
    write_csv(paths["experts"], existing_expert_rows + gptq_expert_rows)
    write_csv(paths["tensors"], existing_tensor_rows + gptq_tensor_rows)
    summary_all, summary_tensors = aggregate_rows(existing_expert_rows + gptq_expert_rows + existing_tensor_rows + gptq_tensor_rows)
    write_summary(paths["summary"], summary_all + summary_tensors)

    metadata_path = args.output_dir / f"{output_stem}_gptq_int4.json"
    metadata = {
        "config": asdict(config),
        "method": METHOD,
        "expert_offset": args.expert_offset,
        "expert_limit": args.expert_limit,
        "experts_completed": len(gptq_expert_rows),
        "elapsed_seconds": time.time() - started,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"experts_csv={paths['experts']}", flush=True)
    print(f"tensors_csv={paths['tensors']}", flush=True)
    print(f"summary_csv={paths['summary']}", flush=True)
    print(f"metadata_json={metadata_path}", flush=True)
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

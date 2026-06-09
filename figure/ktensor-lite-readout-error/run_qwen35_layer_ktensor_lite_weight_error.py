#!/usr/bin/env python3
"""Compare ktensor-lite direct quantization with 8-bit progressive readout.

The metric is weight-space relative L2 error against the original BF16 model
weights:

    ||decoded_weight - original_weight||_2 / ||original_weight||_2

By default the progressive 8-bit object is read from the PagedMoE runtime
storage, which matches the deployed scattered-lite readout path. Use
``--readout-source reencode`` to create the 8-bit ktensor-lite object directly
from the original weights inside this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SUBEXP_DIR = REPO_ROOT / "subexp"
if str(SUBEXP_DIR) not in sys.path:
    sys.path.insert(0, str(SUBEXP_DIR))

from pagedmoe_ktensor_lite_io import (  # noqa: E402
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    PagedMoeExpertReader,
    load_qwen35_expert_weights,
)


DEFAULT_MODEL_ROOT = Path("/mnt/data/Qwen3.5-397B-A17B")
DEFAULT_STORAGE_ROOT = Path("/mnt/test/qwen35_397b_runtime_sharded_rust")
DEFAULT_KTENSOR_LIB_DIR = REPO_ROOT / "src" / "target" / "release"
DEFAULT_LAYER_INDEX = 30
DEFAULT_OUTPUT_STEM = "qwen35_layer30_ktensor_lite_weight_error"
DEFAULT_BITS = tuple(range(1, 9))
TENSOR_ORDER = ("gate", "up", "down")
BLOCK_ELEMENTS = 128 * 1024


@dataclass(frozen=True)
class WorkerConfig:
    model_root: str
    storage_root: str
    ktensor_lib_dir: str
    layer_index: int
    bits: tuple[int, ...]
    sampling_rate: float
    readout_source: str
    torch_threads: int


@dataclass
class Accumulator:
    num_weights: int = 0
    diff_l2_sq: float = 0.0
    reference_l2_sq: float = 0.0
    abs_error_sum: float = 0.0
    max_absolute_error: float = 0.0

    def add(self, reference: torch.Tensor, decoded_bf16: torch.Tensor, reference_l2_sq: float) -> None:
        diff = decoded_bf16.float() - reference
        self.num_weights += int(reference.numel())
        self.diff_l2_sq += float(torch.dot(diff, diff).item())
        self.reference_l2_sq += reference_l2_sq
        self.abs_error_sum += float(diff.abs().sum().item())
        max_abs = float(diff.abs().max().item())
        if max_abs > self.max_absolute_error:
            self.max_absolute_error = max_abs

    def merge_row(self, row: dict[str, str]) -> None:
        self.num_weights += int(row["num_weights"])
        self.diff_l2_sq += float(row["diff_l2_sq"])
        self.reference_l2_sq += float(row["reference_l2_sq"])
        self.abs_error_sum += float(row["abs_error_sum"])
        self.max_absolute_error = max(
            self.max_absolute_error, float(row["max_absolute_error"])
        )

    def metrics(self) -> dict[str, float | int]:
        l2_error = math.sqrt(max(self.diff_l2_sq, 0.0))
        reference_l2_norm = math.sqrt(max(self.reference_l2_sq, 0.0))
        denom = max(reference_l2_norm, 1e-12)
        count = max(self.num_weights, 1)
        return {
            "num_weights": self.num_weights,
            "relative_l2_error": l2_error / denom,
            "l2_error": l2_error,
            "reference_l2_norm": reference_l2_norm,
            "rmse": math.sqrt(max(self.diff_l2_sq, 0.0) / count),
            "mean_absolute_error": self.abs_error_sum / count,
            "max_absolute_error": self.max_absolute_error,
            "diff_l2_sq": self.diff_l2_sq,
            "reference_l2_sq": self.reference_l2_sq,
            "abs_error_sum": self.abs_error_sum,
        }


@dataclass
class ExpertResult:
    expert_id: int
    expert_rows: list[dict[str, Any]] = field(default_factory=list)
    tensor_rows: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--ktensor-lib-dir", type=Path, default=DEFAULT_KTENSOR_LIB_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--expert-offset", type=int, default=0)
    parser.add_argument("--expert-limit", type=int, default=512)
    parser.add_argument("--bits", default=",".join(str(bit) for bit in DEFAULT_BITS))
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=0.1,
        help="ktensor-lite sampling rate for direct/reencode quantization; PagedMoE export uses Rust default 0.1",
    )
    parser.add_argument(
        "--readout-source",
        choices=("storage", "reencode"),
        default="storage",
        help="source for the 8-bit object used by progressive readout",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append missing experts to existing per-expert CSV files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing output files instead of refusing to run",
    )
    return parser.parse_args()


def parse_bits(value: str) -> tuple[int, ...]:
    bits = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not bits:
        raise ValueError("--bits must contain at least one integer")
    for bit in bits:
        if bit < 1 or bit > 8:
            raise ValueError(f"bit level {bit} must be in [1, 8]")
    return bits


def import_ktensor(lib_dir: str):
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    import libktensor_python as kt  # noqa: PLC0415

    return kt


def make_codebook_holder(kt):
    dummy = np.zeros(BLOCK_ELEMENTS, dtype=np.float32)
    _, codebook = kt.lite.LiteEncoder(8, 0.1).encode(dummy)
    view = memoryview(codebook)
    if len(view) != 1024:
        raise RuntimeError(f"unexpected ktensor-lite codebook buffer length {len(view)}")
    return codebook, view


def tensor_blocks(tensor: torch.Tensor, tensor_name: str) -> tuple[torch.Tensor, ...]:
    if tensor_name in ("gate", "up"):
        cols = HIDDEN_SIZE
    elif tensor_name == "down":
        cols = MOE_INTERMEDIATE_SIZE
    else:
        raise ValueError(f"unknown tensor {tensor_name!r}")
    if tensor.numel() % BLOCK_ELEMENTS != 0:
        raise ValueError(
            f"{tensor_name} has {tensor.numel()} elements, not divisible by {BLOCK_ELEMENTS}"
        )
    rows_per_block = BLOCK_ELEMENTS // cols
    if tensor.shape[-1] != cols:
        raise ValueError(f"{tensor_name} has unexpected shape {tuple(tensor.shape)}")
    return tuple(
        tensor[start : start + rows_per_block, :].float().contiguous().view(-1)
        for start in range(0, tensor.shape[0], rows_per_block)
    )


def decode_contiguous(kt, bit: int, codebook, encoded, n: int, output: torch.Tensor) -> None:
    encoded_view = np.frombuffer(encoded, dtype=np.uint8)
    kt.lite.decode(
        bit,
        codebook,
        int(encoded_view.ctypes.data),
        n,
        n // 8,
        int(output.data_ptr()),
    )


def decode_storage(
    kt,
    bit: int,
    codebook_holder,
    bitplanes: list[np.ndarray],
    n: int,
    output: torch.Tensor,
) -> None:
    ptrs = np.array([plane.ctypes.data for plane in bitplanes[:bit]], dtype=np.uintp)
    kt.lite.decode_ptrs(bit, codebook_holder, ptrs, n, int(output.data_ptr()))


def method_name(readout_source: str) -> str:
    if readout_source == "storage":
        return "readout_from_pagedmoe_8bit"
    if readout_source == "reencode":
        return "readout_from_reencoded_8bit"
    raise ValueError(f"unknown readout source {readout_source!r}")


def row_from_accumulator(
    *,
    config: WorkerConfig,
    expert_id: int | str,
    tensor: str,
    bit: int,
    method: str,
    accumulator: Accumulator,
    elapsed_seconds: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "layer_index": config.layer_index,
        "expert_id": expert_id,
        "tensor": tensor,
        "bit": bit,
        "method": method,
        "readout_source": config.readout_source,
        "sampling_rate": config.sampling_rate,
        "elapsed_seconds": elapsed_seconds,
    }
    row.update(accumulator.metrics())
    return row


def process_expert(expert_id: int, config_dict: dict[str, Any]) -> ExpertResult:
    config = WorkerConfig(**config_dict)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(max(1, config.torch_threads))
    torch.set_grad_enabled(False)

    kt = import_ktensor(config.ktensor_lib_dir)
    direct_encoders = {
        bit: kt.lite.LiteEncoder(bit, config.sampling_rate) for bit in config.bits
    }
    encoder8 = kt.lite.LiteEncoder(8, config.sampling_rate)
    readout_method = method_name(config.readout_source)
    output = torch.empty(BLOCK_ELEMENTS, dtype=torch.bfloat16)

    expert_accums: dict[tuple[str, int], Accumulator] = {
        ("direct", bit): Accumulator() for bit in config.bits
    }
    expert_accums.update(
        {(readout_method, bit): Accumulator() for bit in config.bits}
    )
    tensor_accums: dict[tuple[str, str, int], Accumulator] = {}
    for tensor_name in TENSOR_ORDER:
        for bit in config.bits:
            tensor_accums[(tensor_name, "direct", bit)] = Accumulator()
            tensor_accums[(tensor_name, readout_method, bit)] = Accumulator()

    reader = None
    codebook_holder = None
    codebook_view = None
    if config.readout_source == "storage":
        reader = PagedMoeExpertReader(config.storage_root)
        codebook_holder, codebook_view = make_codebook_holder(kt)

    started = time.time()
    try:
        gate, up, down = load_qwen35_expert_weights(
            config.model_root, config.layer_index, expert_id
        )
        tensors = {"gate": gate, "up": up, "down": down}

        for tensor_name in TENSOR_ORDER:
            for block_index, reference in enumerate(tensor_blocks(tensors[tensor_name], tensor_name)):
                reference_l2_sq = float(torch.dot(reference, reference).item())
                reference_np = reference.numpy()

                encoded8 = None
                codebook8 = None
                storage_bitplanes = None
                if config.readout_source == "reencode":
                    encoded8, codebook8 = encoder8.encode(reference_np)
                else:
                    assert reader is not None
                    assert codebook_view is not None
                    codebook_view[:] = reader.read_codebook(
                        config.layer_index, expert_id, tensor_name, block_index
                    )
                    storage_bitplanes = reader.read_block_bitplanes(
                        config.layer_index, expert_id, tensor_name, block_index
                    )

                for bit in config.bits:
                    if bit == 8 and encoded8 is not None and codebook8 is not None:
                        direct_encoded = encoded8
                        direct_codebook = codebook8
                    else:
                        direct_encoded, direct_codebook = direct_encoders[bit].encode(reference_np)
                    decode_contiguous(
                        kt, bit, direct_codebook, direct_encoded, reference.numel(), output
                    )
                    expert_accums[("direct", bit)].add(reference, output, reference_l2_sq)
                    tensor_accums[(tensor_name, "direct", bit)].add(
                        reference, output, reference_l2_sq
                    )

                    if config.readout_source == "reencode":
                        assert encoded8 is not None and codebook8 is not None
                        decode_contiguous(kt, bit, codebook8, encoded8, reference.numel(), output)
                    else:
                        assert codebook_holder is not None
                        assert storage_bitplanes is not None
                        decode_storage(
                            kt,
                            bit,
                            codebook_holder,
                            storage_bitplanes,
                            reference.numel(),
                            output,
                        )
                    expert_accums[(readout_method, bit)].add(reference, output, reference_l2_sq)
                    tensor_accums[(tensor_name, readout_method, bit)].add(
                        reference, output, reference_l2_sq
                    )
    finally:
        if reader is not None:
            reader.close()

    elapsed = time.time() - started
    expert_rows = [
        row_from_accumulator(
            config=config,
            expert_id=expert_id,
            tensor="all",
            bit=bit,
            method=method,
            accumulator=expert_accums[(method, bit)],
            elapsed_seconds=elapsed,
        )
        for method in ("direct", readout_method)
        for bit in config.bits
    ]
    tensor_rows = [
        row_from_accumulator(
            config=config,
            expert_id=expert_id,
            tensor=tensor_name,
            bit=bit,
            method=method,
            accumulator=tensor_accums[(tensor_name, method, bit)],
            elapsed_seconds=elapsed,
        )
        for tensor_name in TENSOR_ORDER
        for method in ("direct", readout_method)
        for bit in config.bits
    ]
    return ExpertResult(
        expert_id=expert_id,
        expert_rows=expert_rows,
        tensor_rows=tensor_rows,
        elapsed_seconds=elapsed,
    )


def output_paths(output_dir: Path, output_stem: str) -> dict[str, Path]:
    return {
        "experts": output_dir / f"{output_stem}_experts.csv",
        "tensors": output_dir / f"{output_stem}_tensors.csv",
        "summary": output_dir / f"{output_stem}_summary.csv",
        "metadata": output_dir / f"{output_stem}.json",
    }


def csv_fieldnames() -> list[str]:
    return [
        "layer_index",
        "expert_id",
        "tensor",
        "bit",
        "method",
        "readout_source",
        "sampling_rate",
        "num_weights",
        "relative_l2_error",
        "l2_error",
        "reference_l2_norm",
        "rmse",
        "mean_absolute_error",
        "max_absolute_error",
        "diff_l2_sq",
        "reference_l2_sq",
        "abs_error_sum",
        "elapsed_seconds",
    ]


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def completed_experts(rows: list[dict[str, str]], bits: tuple[int, ...]) -> set[int]:
    expected = {("direct", bit) for bit in bits}
    methods = {row["method"] for row in rows if row["tensor"] == "all"}
    readout_methods = [method for method in methods if method.startswith("readout_from_")]
    if len(readout_methods) != 1:
        return set()
    expected.update({(readout_methods[0], bit) for bit in bits})

    seen: dict[int, set[tuple[str, int]]] = {}
    for row in rows:
        if row["tensor"] != "all":
            continue
        try:
            expert_id = int(row["expert_id"])
            bit = int(row["bit"])
        except ValueError:
            continue
        seen.setdefault(expert_id, set()).add((row["method"], bit))
    return {expert_id for expert_id, values in seen.items() if expected <= values}


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames(), lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_accums: dict[tuple[str, int], Accumulator] = {}
    tensor_accums: dict[tuple[str, str, int], Accumulator] = {}
    layer_index = ""
    readout_source = ""
    sampling_rate = ""

    for row in rows:
        if row["tensor"] != "all":
            continue
        layer_index = row["layer_index"]
        readout_source = row["readout_source"]
        sampling_rate = row["sampling_rate"]
        key = (row["method"], int(row["bit"]))
        all_accums.setdefault(key, Accumulator()).merge_row(row)

    for row in rows:
        if row["tensor"] == "all":
            continue
        key = (row["tensor"], row["method"], int(row["bit"]))
        tensor_accums.setdefault(key, Accumulator()).merge_row(row)

    all_rows = [
        summary_row(
            layer_index=layer_index,
            tensor="all",
            bit=bit,
            method=method,
            readout_source=readout_source,
            sampling_rate=sampling_rate,
            accumulator=accumulator,
        )
        for (method, bit), accumulator in sorted(all_accums.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    tensor_rows = [
        summary_row(
            layer_index=layer_index,
            tensor=tensor,
            bit=bit,
            method=method,
            readout_source=readout_source,
            sampling_rate=sampling_rate,
            accumulator=accumulator,
        )
        for (tensor, method, bit), accumulator in sorted(
            tensor_accums.items(), key=lambda item: (TENSOR_ORDER.index(item[0][0]), item[0][1], item[0][2])
        )
    ]
    return all_rows, tensor_rows


def summary_row(
    *,
    layer_index: str,
    tensor: str,
    bit: int,
    method: str,
    readout_source: str,
    sampling_rate: str,
    accumulator: Accumulator,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "layer_index": layer_index,
        "expert_id": "all",
        "tensor": tensor,
        "bit": bit,
        "method": method,
        "readout_source": readout_source,
        "sampling_rate": sampling_rate,
        "elapsed_seconds": "",
    }
    row.update(accumulator.metrics())
    return row


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_outputs(paths: dict[str, Path], *, resume: bool, force: bool) -> None:
    paths["experts"].parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths.values() if path.exists()]
    if not existing:
        return
    if resume:
        return
    if force:
        for path in existing:
            path.unlink()
        return
    names = ", ".join(str(path) for path in existing)
    raise FileExistsError(f"output files already exist; pass --resume or --force: {names}")


def run(args: argparse.Namespace) -> int:
    bits = parse_bits(args.bits)
    if args.expert_offset < 0 or args.expert_limit < 1:
        raise ValueError("--expert-offset must be non-negative and --expert-limit positive")
    expert_end = min(512, args.expert_offset + args.expert_limit)
    expert_ids = list(range(args.expert_offset, expert_end))
    if not expert_ids:
        raise ValueError("empty expert range")
    if args.readout_source == "storage" and not (args.storage_root / "manifest.json").exists():
        raise FileNotFoundError(f"PagedMoE manifest not found under {args.storage_root}")

    output_stem = args.output_stem
    output_stem_defaulted = output_stem == DEFAULT_OUTPUT_STEM
    if args.layer_index != DEFAULT_LAYER_INDEX and output_stem_defaulted:
        output_stem = f"qwen35_layer{args.layer_index}_ktensor_lite_weight_error"
    if args.readout_source != "storage" and output_stem_defaulted:
        output_stem = f"{output_stem}_{args.readout_source}"
    paths = output_paths(args.output_dir, output_stem)
    prepare_outputs(paths, resume=args.resume, force=args.force)

    existing_expert_rows = read_existing_rows(paths["experts"]) if args.resume else []
    existing_tensor_rows = read_existing_rows(paths["tensors"]) if args.resume else []
    done = completed_experts(existing_expert_rows, bits) if args.resume else set()
    pending_experts = [expert_id for expert_id in expert_ids if expert_id not in done]

    config = WorkerConfig(
        model_root=str(args.model_root),
        storage_root=str(args.storage_root),
        ktensor_lib_dir=str(args.ktensor_lib_dir),
        layer_index=args.layer_index,
        bits=bits,
        sampling_rate=args.sampling_rate,
        readout_source=args.readout_source,
        torch_threads=args.torch_threads,
    )

    metadata = {
        "description": __doc__,
        "config": asdict(config),
        "expert_offset": args.expert_offset,
        "expert_limit": args.expert_limit,
        "experts_requested": len(expert_ids),
        "experts_completed_before_run": len(done),
        "experts_pending_at_start": len(pending_experts),
        "workers": args.workers,
        "output_files": {key: str(path) for key, path in paths.items()},
        "started_unix": time.time(),
    }

    print(
        f"layer={args.layer_index} experts={args.expert_offset}..{expert_end - 1} "
        f"pending={len(pending_experts)} bits={bits} readout_source={args.readout_source} "
        f"workers={args.workers}",
        flush=True,
    )

    run_started = time.time()
    completed_now = 0
    config_dict = asdict(config)
    if args.workers <= 1:
        for expert_id in pending_experts:
            result = process_expert(expert_id, config_dict)
            append_rows(paths["experts"], result.expert_rows)
            append_rows(paths["tensors"], result.tensor_rows)
            completed_now += 1
            print(
                f"expert={expert_id} elapsed={result.elapsed_seconds:.2f}s "
                f"progress={completed_now}/{len(pending_experts)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_expert, expert_id, config_dict): expert_id
                for expert_id in pending_experts
            }
            for future in as_completed(futures):
                expert_id = futures[future]
                result = future.result()
                append_rows(paths["experts"], result.expert_rows)
                append_rows(paths["tensors"], result.tensor_rows)
                completed_now += 1
                print(
                    f"expert={expert_id} elapsed={result.elapsed_seconds:.2f}s "
                    f"progress={completed_now}/{len(pending_experts)}",
                    flush=True,
                )

    all_expert_rows = read_existing_rows(paths["experts"])
    all_tensor_rows = read_existing_rows(paths["tensors"])
    summary_all, summary_tensors = aggregate_rows(all_expert_rows + all_tensor_rows)
    write_summary(paths["summary"], summary_all + summary_tensors)

    metadata.update(
        {
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - run_started,
            "experts_completed_this_run": completed_now,
            "experts_completed_total": len(completed_experts(all_expert_rows, bits)),
        }
    )
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"experts_csv={paths['experts']}", flush=True)
    print(f"tensors_csv={paths['tensors']}", flush=True)
    print(f"summary_csv={paths['summary']}", flush=True)
    print(f"metadata_json={paths['metadata']}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

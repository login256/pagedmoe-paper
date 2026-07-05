#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mmap
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from subexp.pagedmoe_ktensor_lite_io import (
    TENSORS_PER_EXPERT,
    layer_bitplanes_path,
    layer_codebooks_path,
    load_manifest,
    load_qwen35_expert_weights,
)


DEFAULT_STORAGE_ROOT = Path("/mnt/test/qwen35_397b_runtime_full_rust_fixed")
DEFAULT_MODEL_ROOT = Path("/mnt/data/Qwen3.5-397B-A17B")
DEFAULT_PACK_ROOT = Path("/mnt/test/qwen35_397b_10expert_int_pack")


@dataclass
class DenseExpert:
    expert_id: int
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass
class W8A16Expert:
    expert_id: int
    gate: nn.Linear
    up: nn.Linear
    down: nn.Linear


@dataclass
class ReadResult:
    expert_id: int
    start_ns: int
    end_ns: int
    codebook_start_ns: int
    codebook_end_ns: int
    bitplane_start_ns: int
    bitplane_end_ns: int
    bytes_read: int


@dataclass
class CompactReadResult:
    expert_ids: tuple[int, ...]
    start_ns: int
    end_ns: int
    bytes_read: int
    offset: int
    length: int


def ns() -> int:
    return time.perf_counter_ns()


def ms(delta_ns: int) -> float:
    return delta_ns / 1_000_000.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def cpulist_to_set(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def bind_numa_node(node: int | None) -> int | None:
    if node is None:
        return None
    cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not cpulist.exists():
        raise SystemExit(f"NUMA node {node} does not exist: {cpulist}")
    cpus = cpulist_to_set(cpulist.read_text())
    os.sched_setaffinity(0, cpus)
    return len(cpus)


def drop_fd_cache(fd: int, offset: int, length: int) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((s, e) for s, e in intervals if e > s)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(e - s for s, e in merge_intervals(intervals))


def interval_overlap(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> int:
    aa = merge_intervals(a)
    bb = merge_intervals(b)
    i = j = total = 0
    while i < len(aa) and j < len(bb):
        start = max(aa[i][0], bb[j][0])
        end = min(aa[i][1], bb[j][1])
        if end > start:
            total += end - start
        if aa[i][1] < bb[j][1]:
            i += 1
        else:
            j += 1
    return total


class Int4RuntimeReader:
    def __init__(self, storage_root: Path, layer_index: int, precision_bits: int):
        self.storage_root = storage_root
        self.layer_index = layer_index
        self.manifest = load_manifest(storage_root)
        if self.manifest.bitplane_shard_count != 1:
            raise SystemExit("this benchmark currently expects unsharded bitplanes")
        if precision_bits < 1 or precision_bits > self.manifest.max_logk:
            raise SystemExit(f"invalid precision_bits={precision_bits}")
        self.precision_bits = precision_bits
        self.codebook_path = layer_codebooks_path(storage_root, layer_index)
        self.bitplane_path = layer_bitplanes_path(storage_root, self.manifest, layer_index, 0)
        self.codebook_fd = os.open(self.codebook_path, os.O_RDONLY)
        self.bitplane_fd = os.open(self.bitplane_path, os.O_RDONLY)
        self.codebook_per_expert = (
            TENSORS_PER_EXPERT * self.manifest.num_blocks * self.manifest.codebook_bytes
        )
        self.bitplane_per_expert_plane = (
            TENSORS_PER_EXPERT * self.manifest.num_blocks * self.manifest.bitplane_bytes
        )
        self.bytes_per_expert = (
            self.codebook_per_expert
            + self.precision_bits * self.bitplane_per_expert_plane
        )

    def close(self) -> None:
        os.close(self.codebook_fd)
        os.close(self.bitplane_fd)

    def __enter__(self) -> "Int4RuntimeReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_expert_bytes(
        self,
        expert_id: int,
        drop_cache: bool,
    ) -> tuple[bytes, bytes, tuple[int, int, int, int, int, int]]:
        codebook_offset = expert_id * self.codebook_per_expert
        bitplane_offset = expert_id * self.manifest.max_logk * self.bitplane_per_expert_plane
        bitplane_len = self.precision_bits * self.bitplane_per_expert_plane

        if drop_cache:
            drop_fd_cache(self.codebook_fd, codebook_offset, self.codebook_per_expert)
            drop_fd_cache(self.bitplane_fd, bitplane_offset, bitplane_len)

        start = ns()
        codebook_start = ns()
        codebooks = os.pread(self.codebook_fd, self.codebook_per_expert, codebook_offset)
        codebook_end = ns()
        if len(codebooks) != self.codebook_per_expert:
            raise OSError(f"short codebook read: {len(codebooks)} bytes")

        bitplane_start = ns()
        bitplanes = os.pread(self.bitplane_fd, bitplane_len, bitplane_offset)
        bitplane_end = ns()
        if len(bitplanes) != bitplane_len:
            raise OSError(f"short bitplane read: {len(bitplanes)} bytes")
        end = ns()
        return codebooks, bitplanes, (start, codebook_start, codebook_end, bitplane_start, bitplane_end, end)

    def read_expert(self, expert_id: int, drop_cache: bool) -> ReadResult:
        codebooks, bitplanes, times = self.read_expert_bytes(expert_id, drop_cache)
        start, codebook_start, codebook_end, bitplane_start, bitplane_end, end = times
        return ReadResult(
            expert_id=expert_id,
            start_ns=start,
            end_ns=end,
            codebook_start_ns=codebook_start,
            codebook_end_ns=codebook_end,
            bitplane_start_ns=bitplane_start,
            bitplane_end_ns=bitplane_end,
            bytes_read=len(codebooks) + len(bitplanes),
        )


def expert_id_span_label(expert_ids: list[int]) -> str:
    if expert_ids == list(range(expert_ids[0], expert_ids[-1] + 1)):
        return f"{expert_ids[0]:03d}_{expert_ids[-1]:03d}"
    return "selected_" + "_".join(f"{value:03d}" for value in expert_ids)


def default_pack_file(layer_index: int, expert_ids: list[int], precision_bits: int) -> Path:
    name = (
        f"qwen35_397b_layer_{layer_index:03d}_experts_"
        f"{expert_id_span_label(expert_ids)}_int{precision_bits}.bin"
    )
    return DEFAULT_PACK_ROOT / name


def default_w8a16_pack_file(layer_index: int, expert_ids: list[int]) -> Path:
    name = (
        f"qwen35_397b_layer_{layer_index:03d}_experts_"
        f"{expert_id_span_label(expert_ids)}_w8a16.bin"
    )
    return DEFAULT_PACK_ROOT / name


def default_bf16_pack_file(layer_index: int, expert_ids: list[int]) -> Path:
    name = (
        f"qwen35_397b_layer_{layer_index:03d}_experts_"
        f"{expert_id_span_label(expert_ids)}_bf16.bin"
    )
    return DEFAULT_PACK_ROOT / name


def compact_ranges(indexes: list[int], per_expert_bytes: int) -> list[tuple[int, int]]:
    if not indexes:
        return []
    ordered = sorted(indexes)
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for index in ordered[1:]:
        if index == prev + 1:
            prev = index
            continue
        ranges.append((start * per_expert_bytes, (prev - start + 1) * per_expert_bytes))
        start = prev = index
    ranges.append((start * per_expert_bytes, (prev - start + 1) * per_expert_bytes))
    return ranges


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def ensure_compact_pack(
    source_reader: Int4RuntimeReader,
    pack_file: Path,
    layer_index: int,
    expert_ids: list[int],
    repack: bool,
) -> Path:
    metadata_file = pack_file.with_suffix(pack_file.suffix + ".json")
    expected = {
        "format": f"qwen35_397b_compact_expert_int{source_reader.precision_bits}_v1",
        "source_storage_root": str(source_reader.storage_root),
        "layer_index": layer_index,
        "expert_ids": expert_ids,
        "precision_bits": source_reader.precision_bits,
        "bytes_per_expert": source_reader.bytes_per_expert,
        "codebook_per_expert": source_reader.codebook_per_expert,
        "bitplane_per_expert_plane": source_reader.bitplane_per_expert_plane,
        "total_bytes": source_reader.bytes_per_expert * len(expert_ids),
    }
    if not repack and pack_file.exists() and metadata_file.exists():
        try:
            actual = json.loads(metadata_file.read_text())
        except json.JSONDecodeError:
            actual = {}
        if (
            actual == expected
            and pack_file.stat().st_size == expected["total_bytes"]
        ):
            return pack_file

    pack_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = pack_file.with_suffix(pack_file.suffix + ".tmp")
    with tmp.open("wb", buffering=0) as f:
        for expert_id in expert_ids:
            codebooks, bitplanes, _ = source_reader.read_expert_bytes(
                expert_id,
                drop_cache=False,
            )
            f.write(codebooks)
            f.write(bitplanes)
    os.replace(tmp, pack_file)
    metadata_file.write_text(json.dumps(expected, indent=2) + "\n")
    return pack_file


def tensor_payload_bytes(tensor: torch.Tensor) -> bytes:
    payload = tensor.detach().cpu().contiguous()
    if payload.dtype == torch.bfloat16:
        return payload.view(torch.uint8).numpy().tobytes()
    return payload.numpy().tobytes()


def bf16_tensor_payload_bytes(tensor: torch.Tensor) -> int:
    if tensor.dtype != torch.bfloat16:
        raise SystemExit(f"expected BF16 tensor payload, got {tensor.dtype}")
    return tensor.numel() * tensor.element_size()


def bf16_expert_payload_bytes(expert: DenseExpert) -> int:
    return (
        bf16_tensor_payload_bytes(expert.gate)
        + bf16_tensor_payload_bytes(expert.up)
        + bf16_tensor_payload_bytes(expert.down)
    )


def write_bf16_tensor_payload(f, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise SystemExit(f"expected BF16 tensor payload, got {tensor.dtype}")
    f.write(tensor_payload_bytes(tensor))


def bf16_tensor_payload_metadata(tensor: torch.Tensor) -> dict[str, object]:
    if tensor.dtype != torch.bfloat16:
        raise SystemExit(f"expected BF16 tensor payload, got {tensor.dtype}")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def ensure_bf16_pack(
    pack_file: Path,
    layer_index: int,
    expert_ids: list[int],
    experts: dict[int, DenseExpert],
    repack: bool,
) -> tuple[Path, int]:
    metadata_file = pack_file.with_suffix(pack_file.suffix + ".json")
    payload_bytes_per_expert = bf16_expert_payload_bytes(experts[expert_ids[0]])
    bytes_per_expert = align_up(
        payload_bytes_per_expert,
        CompactPackReader.direct_io_alignment,
    )
    padding_bytes_per_expert = bytes_per_expert - payload_bytes_per_expert
    sample = experts[expert_ids[0]]
    tensor_layout = {
        "gate": bf16_tensor_payload_metadata(sample.gate),
        "up": bf16_tensor_payload_metadata(sample.up),
        "down": bf16_tensor_payload_metadata(sample.down),
    }
    expected = {
        "format": "qwen35_397b_compact_expert_bf16_raw_v1",
        "layer_index": layer_index,
        "expert_ids": expert_ids,
        "weight_dtype": "bfloat16",
        "activation_dtype": "bfloat16",
        "bytes_per_expert": bytes_per_expert,
        "payload_bytes_per_expert": payload_bytes_per_expert,
        "padding_bytes_per_expert": padding_bytes_per_expert,
        "tensor_layout": tensor_layout,
        "total_bytes": bytes_per_expert * len(expert_ids),
    }
    if not repack and pack_file.exists() and metadata_file.exists():
        try:
            actual = json.loads(metadata_file.read_text())
        except json.JSONDecodeError:
            actual = {}
        if actual == expected and pack_file.stat().st_size == expected["total_bytes"]:
            return pack_file, bytes_per_expert

    pack_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = pack_file.with_suffix(pack_file.suffix + ".tmp")
    with tmp.open("wb", buffering=0) as f:
        for expert_id in expert_ids:
            expert = experts[expert_id]
            start = f.tell()
            write_bf16_tensor_payload(f, expert.gate)
            write_bf16_tensor_payload(f, expert.up)
            write_bf16_tensor_payload(f, expert.down)
            written = f.tell() - start
            if written != payload_bytes_per_expert:
                raise SystemExit(
                    f"expert {expert_id} BF16 payload changed size: "
                    f"{written} != {payload_bytes_per_expert}"
                )
            if padding_bytes_per_expert:
                f.write(b"\0" * padding_bytes_per_expert)
    os.replace(tmp, pack_file)
    metadata_file.write_text(json.dumps(expected, indent=2) + "\n")
    return pack_file, bytes_per_expert


def w8a16_linear_payload_tensors(linear: nn.Linear) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = linear.weight
    tensor_impl = getattr(weight, "tensor_impl", None)
    if tensor_impl is None:
        raise SystemExit("expected torchao AffineQuantizedTensor weight with tensor_impl")
    data = getattr(tensor_impl, "data", None)
    scale = getattr(tensor_impl, "scale", None)
    zero_point = getattr(tensor_impl, "zero_point", None)
    if data is None or scale is None or zero_point is None:
        raise SystemExit("torchao W8A16 weight is missing data/scale/zero_point tensors")
    data = getattr(data, "int_data", data)
    if data.dtype != torch.int8:
        raise SystemExit(f"expected int8 weight payload, got {data.dtype}")
    return data, scale, zero_point


def w8a16_linear_payload_bytes(linear: nn.Linear) -> int:
    return sum(t.numel() * t.element_size() for t in w8a16_linear_payload_tensors(linear))


def w8a16_expert_payload_bytes(expert: W8A16Expert) -> int:
    return (
        w8a16_linear_payload_bytes(expert.gate)
        + w8a16_linear_payload_bytes(expert.up)
        + w8a16_linear_payload_bytes(expert.down)
    )


def write_w8a16_linear_payload(f, linear: nn.Linear) -> dict[str, object]:
    data, scale, zero_point = w8a16_linear_payload_tensors(linear)
    f.write(tensor_payload_bytes(data))
    f.write(tensor_payload_bytes(scale))
    f.write(tensor_payload_bytes(zero_point))
    return {
        "data_shape": list(data.shape),
        "data_dtype": str(data.dtype),
        "scale_shape": list(scale.shape),
        "scale_dtype": str(scale.dtype),
        "zero_point_shape": list(zero_point.shape),
        "zero_point_dtype": str(zero_point.dtype),
        "bytes": data.numel() * data.element_size()
        + scale.numel() * scale.element_size()
        + zero_point.numel() * zero_point.element_size(),
    }


def ensure_w8a16_pack(
    pack_file: Path,
    layer_index: int,
    expert_ids: list[int],
    experts: dict[int, W8A16Expert],
    repack: bool,
) -> tuple[Path, int]:
    metadata_file = pack_file.with_suffix(pack_file.suffix + ".json")
    payload_bytes_per_expert = w8a16_expert_payload_bytes(experts[expert_ids[0]])
    bytes_per_expert = align_up(
        payload_bytes_per_expert,
        CompactPackReader.direct_io_alignment,
    )
    padding_bytes_per_expert = bytes_per_expert - payload_bytes_per_expert
    linear_layout = {
        "gate": write_w8a16_linear_payload_metadata(experts[expert_ids[0]].gate),
        "up": write_w8a16_linear_payload_metadata(experts[expert_ids[0]].up),
        "down": write_w8a16_linear_payload_metadata(experts[expert_ids[0]].down),
    }
    expected = {
        "format": "qwen35_397b_compact_expert_w8a16_torchao_v1",
        "layer_index": layer_index,
        "expert_ids": expert_ids,
        "weight_dtype": "int8",
        "activation_dtype": "bfloat16",
        "torchao_config": "Int8WeightOnlyConfig",
        "bytes_per_expert": bytes_per_expert,
        "payload_bytes_per_expert": payload_bytes_per_expert,
        "padding_bytes_per_expert": padding_bytes_per_expert,
        "linear_layout": linear_layout,
        "total_bytes": bytes_per_expert * len(expert_ids),
    }
    if not repack and pack_file.exists() and metadata_file.exists():
        try:
            actual = json.loads(metadata_file.read_text())
        except json.JSONDecodeError:
            actual = {}
        if actual == expected and pack_file.stat().st_size == expected["total_bytes"]:
            return pack_file, bytes_per_expert

    pack_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = pack_file.with_suffix(pack_file.suffix + ".tmp")
    with tmp.open("wb", buffering=0) as f:
        for expert_id in expert_ids:
            expert = experts[expert_id]
            start = f.tell()
            for linear in (expert.gate, expert.up, expert.down):
                write_w8a16_linear_payload(f, linear)
            written = f.tell() - start
            if written != payload_bytes_per_expert:
                raise SystemExit(
                    f"expert {expert_id} W8A16 payload changed size: "
                    f"{written} != {payload_bytes_per_expert}"
                )
            if padding_bytes_per_expert:
                f.write(b"\0" * padding_bytes_per_expert)
    os.replace(tmp, pack_file)
    metadata_file.write_text(json.dumps(expected, indent=2) + "\n")
    return pack_file, bytes_per_expert


def write_w8a16_linear_payload_metadata(linear: nn.Linear) -> dict[str, object]:
    data, scale, zero_point = w8a16_linear_payload_tensors(linear)
    return {
        "data_shape": list(data.shape),
        "data_dtype": str(data.dtype),
        "data_bytes": data.numel() * data.element_size(),
        "scale_shape": list(scale.shape),
        "scale_dtype": str(scale.dtype),
        "scale_bytes": scale.numel() * scale.element_size(),
        "zero_point_shape": list(zero_point.shape),
        "zero_point_dtype": str(zero_point.dtype),
        "zero_point_bytes": zero_point.numel() * zero_point.element_size(),
    }


class CompactPackReader:
    direct_io_alignment = 4096

    def __init__(self, pack_file: Path, expert_ids: list[int], bytes_per_expert: int):
        if not hasattr(os, "O_DIRECT"):
            raise SystemExit("O_DIRECT is required for this benchmark, but os.O_DIRECT is unavailable")
        if not hasattr(os, "preadv"):
            raise SystemExit("os.preadv is required for O_DIRECT reads")
        if bytes_per_expert % self.direct_io_alignment != 0:
            raise SystemExit(
                f"bytes_per_expert={bytes_per_expert} is not {self.direct_io_alignment}-byte aligned"
            )
        self.pack_file = pack_file
        self.expert_ids = tuple(expert_ids)
        self.index_by_expert = {expert_id: index for index, expert_id in enumerate(expert_ids)}
        self.bytes_per_expert = bytes_per_expert
        self.buffer_size = len(expert_ids) * bytes_per_expert
        if pack_file.stat().st_size % self.direct_io_alignment != 0:
            raise SystemExit(
                f"pack_file size={pack_file.stat().st_size} is not {self.direct_io_alignment}-byte aligned"
            )
        try:
            self.fd = os.open(pack_file, os.O_RDONLY | os.O_DIRECT)
        except OSError as exc:
            raise SystemExit(f"failed to open {pack_file} with O_DIRECT: {exc}") from exc
        self.direct_io = True
        self.buffer = mmap.mmap(-1, self.buffer_size)

    def close(self) -> None:
        self.buffer.close()
        os.close(self.fd)

    def __enter__(self) -> "CompactPackReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_experts(
        self,
        expert_ids: list[int],
        drop_cache: bool,
        start_event: Event | None = None,
        ready_event: Event | None = None,
    ) -> CompactReadResult:
        indexes = [self.index_by_expert[expert_id] for expert_id in expert_ids]
        ranges = compact_ranges(indexes, self.bytes_per_expert)
        if drop_cache and not self.direct_io:
            for offset, length in ranges:
                drop_fd_cache(self.fd, offset, length)

        if ready_event is not None:
            ready_event.set()
        if start_event is not None:
            start_event.wait()

        start = ns()
        bytes_read = 0
        first_offset = ranges[0][0] if ranges else 0
        total_length = 0
        buffer_pos = 0
        for offset, length in ranges:
            view = memoryview(self.buffer)[buffer_pos : buffer_pos + length]
            try:
                got = os.preadv(self.fd, [view], offset)
            finally:
                view.release()
            if got != length:
                raise OSError(f"short compact pack read: {got} bytes, expected {length}")
            bytes_read += got
            total_length += length
            buffer_pos += length
        end = ns()
        return CompactReadResult(
            expert_ids=tuple(expert_ids),
            start_ns=start,
            end_ns=end,
            bytes_read=bytes_read,
            offset=first_offset,
            length=total_length,
        )


def load_dense_experts(model_root: Path, layer_index: int, expert_ids: list[int]) -> dict[int, DenseExpert]:
    experts: dict[int, DenseExpert] = {}
    for expert_id in expert_ids:
        gate, up, down = load_qwen35_expert_weights(model_root, layer_index, expert_id)
        experts[expert_id] = DenseExpert(expert_id, gate.contiguous(), up.contiguous(), down.contiguous())
    return experts


def make_w8a16_linear(weight: torch.Tensor) -> nn.Linear:
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    linear = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        dtype=weight.dtype,
        device="cpu",
    ).eval()
    with torch.no_grad():
        linear.weight.copy_(weight.contiguous())
    quantize_(linear, Int8WeightOnlyConfig())
    data, _, _ = w8a16_linear_payload_tensors(linear)
    if data.dtype != torch.int8:
        raise SystemExit(f"torchao W8A16 quantization did not produce int8 data: {data.dtype}")
    return linear


def load_w8a16_experts(model_root: Path, layer_index: int, expert_ids: list[int]) -> dict[int, W8A16Expert]:
    experts: dict[int, W8A16Expert] = {}
    for expert_id in expert_ids:
        gate, up, down = load_qwen35_expert_weights(model_root, layer_index, expert_id)
        experts[expert_id] = W8A16Expert(
            expert_id=expert_id,
            gate=make_w8a16_linear(gate),
            up=make_w8a16_linear(up),
            down=make_w8a16_linear(down),
        )
        del gate, up, down
    return experts


def assert_cpu_experts(experts: dict[int, DenseExpert | W8A16Expert], x: torch.Tensor) -> None:
    if x.device.type != "cpu":
        raise SystemExit(f"input tensor is not on CPU: {x.device}")
    for expert_id, expert in experts.items():
        if isinstance(expert, DenseExpert):
            for name, tensor in (("gate", expert.gate), ("up", expert.up), ("down", expert.down)):
                if tensor.device.type != "cpu":
                    raise SystemExit(f"expert {expert_id} {name} is not on CPU: {tensor.device}")
        else:
            for name, linear in (("gate", expert.gate), ("up", expert.up), ("down", expert.down)):
                if linear.weight.device.type != "cpu":
                    raise SystemExit(f"expert {expert_id} {name} is not on CPU: {linear.weight.device}")
                data, _, _ = w8a16_linear_payload_tensors(linear)
                if data.dtype != torch.int8:
                    raise SystemExit(f"expert {expert_id} {name} is not backed by int8 weight data")


def compute_expert(expert: DenseExpert | W8A16Expert, x: torch.Tensor) -> torch.Tensor:
    if isinstance(expert, DenseExpert):
        gate = F.linear(x, expert.gate)
        up = F.linear(x, expert.up)
        return F.linear(F.silu(gate) * up, expert.down)
    gate = expert.gate(x)
    up = expert.up(x)
    return expert.down(F.silu(gate) * up)


def run_compute_only(
    experts: dict[int, DenseExpert | W8A16Expert],
    x: torch.Tensor,
    expert_ids: list[int],
    repeats: int,
    warmup: int,
) -> tuple[list[float], float]:
    checksum = 0.0
    samples: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            for expert_id in expert_ids:
                checksum += float(compute_expert(experts[expert_id], x)[0, 0])
        for _ in range(repeats):
            start = ns()
            for expert_id in expert_ids:
                checksum += float(compute_expert(experts[expert_id], x)[0, 0])
            end = ns()
            samples.append(ms(end - start))
    return samples, checksum


def read_miss_sequence(
    reader: CompactPackReader,
    miss_ids: list[int],
    drop_cache: bool,
    output_queue: Queue[CompactReadResult | BaseException],
    start_event: Event,
    ready_event: Event,
) -> None:
    try:
        for index, expert_id in enumerate(miss_ids):
            result = reader.read_experts(
                [expert_id],
                drop_cache,
                start_event if index == 0 else None,
                ready_event if index == 0 else None,
            )
            output_queue.put(result)
    except BaseException as exc:
        ready_event.set()
        output_queue.put(exc)
        raise


def run_pipeline_once(
    reader: CompactPackReader,
    loader_pool: ThreadPoolExecutor,
    experts: dict[int, DenseExpert | W8A16Expert],
    hit_ids: list[int],
    miss_ids: list[int],
    x: torch.Tensor,
    drop_cache: bool,
) -> dict[str, float | int | str]:
    read_results: list[CompactReadResult] = []
    compute_intervals: list[tuple[int, int]] = []
    checksum = 0.0
    if miss_ids:
        start_event = Event()
        ready_event = Event()
        read_queue: Queue[CompactReadResult | BaseException] | None = Queue(maxsize=1)
        read_future = (
            loader_pool.submit(
                read_miss_sequence,
                reader,
                miss_ids,
                drop_cache,
                read_queue,
                start_event,
                ready_event,
            )
        )
        ready_event.wait()
        wall_start = ns()
        start_event.set()
    else:
        read_queue = None
        read_future = None
        wall_start = ns()

    with torch.no_grad():
        for expert_id in hit_ids:
            start = ns()
            checksum += float(compute_expert(experts[expert_id], x)[0, 0])
            end = ns()
            compute_intervals.append((start, end))

        if read_queue is not None:
            for expert_id in miss_ids:
                result = read_queue.get()
                if isinstance(result, BaseException):
                    if read_future is not None:
                        read_future.result()
                    raise result
                read_results.append(result)
                start = ns()
                checksum += float(compute_expert(experts[expert_id], x)[0, 0])
                end = ns()
                compute_intervals.append((start, end))
            if read_future is not None:
                read_future.result()
    wall_end = ns()

    read_intervals = [(r.start_ns, r.end_ns) for r in read_results]
    compute_union_ns = interval_duration(compute_intervals)
    read_union_ns = interval_duration(read_intervals)
    read_overlap_ns = interval_overlap(read_intervals, compute_intervals)
    exposed_read_ns = max(0, read_union_ns - read_overlap_ns)
    if read_results:
        read_offset = min(r.offset for r in read_results)
        read_length = max(r.offset + r.length for r in read_results) - read_offset
    else:
        read_offset = 0
        read_length = 0

    return {
        "wall_ms": ms(wall_end - wall_start),
        "compute_sum_ms": ms(sum(e - s for s, e in compute_intervals)),
        "compute_union_ms": ms(compute_union_ns),
        "read_sum_ms": ms(sum(r.end_ns - r.start_ns for r in read_results)),
        "read_union_ms": ms(read_union_ns),
        "read_exposed_ms": ms(exposed_read_ns),
        "read_overlap_ms": ms(read_overlap_ns),
        "bytes_read": sum(r.bytes_read for r in read_results),
        "read_offset": read_offset,
        "read_length": read_length,
        "read_ops": len(read_results),
        "checksum": checksum,
    }


def pct(value: float, baseline: float) -> float:
    return 100.0 * value / baseline if baseline > 0 else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg_plot(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(int(r["hit_experts"])) for r in rows]
    read_exposed = [float(r["read_exposed_pct_of_compute"]) for r in rows]
    read_sum = [float(r["read_sum_pct_of_compute"]) for r in rows]
    wall = [float(r["wall_pct_of_compute"]) for r in rows]
    rel_throughput = [
        float(
            r.get(
                "relative_expert_throughput_pct",
                r.get("relative_expert_compute_util_pct", r["relative_mfu_pct"]),
            )
        )
        for r in rows
    ]
    ymax = max([100.0, *read_exposed, *read_sum, *wall, *rel_throughput]) * 1.12
    width, height = 760, 460
    left, right, top, bottom = 70, 24, 28, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(1, len(labels))
    bar_w = min(36.0, group_w / 4.5)

    def y(value: float) -> float:
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460">',
        '<rect width="760" height="460" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#222}.small{font-size:12px}.axis{stroke:#333;stroke-width:1}.grid{stroke:#D8DEE9;stroke-width:1}</style>',
    ]
    for tick in range(0, int(ymax) + 1, 100):
        yy = y(float(tick))
        parts.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="small" x="{left-10}" y="{yy+4:.1f}" text-anchor="end">{tick}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>')

    colors = {"wall": "#4C78A8", "read": "#54A24B", "exposed": "#F58518", "throughput": "#111111"}
    points: list[tuple[float, float]] = []
    for i, label in enumerate(labels):
        cx = left + group_w * (i + 0.5)
        for offset, value, color in [
            (-bar_w, wall[i], colors["wall"]),
            (0.0, read_sum[i], colors["read"]),
            (bar_w, read_exposed[i], colors["exposed"]),
        ]:
            x0 = cx + offset - bar_w / 2
            yy = y(value)
            parts.append(
                f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{height-bottom-yy:.1f}" fill="{color}"/>'
            )
        points.append((cx, y(rel_throughput[i])))
        parts.append(f'<text x="{cx:.1f}" y="{height-bottom+24}" text-anchor="middle">{label}</text>')
    polyline = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    parts.append(f'<polyline points="{polyline}" fill="none" stroke="{colors["throughput"]}" stroke-width="2"/>')
    for px, py in points:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colors["throughput"]}"/>')

    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height-18}" text-anchor="middle">hit experts out of 10</text>')
    parts.append(
        f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" text-anchor="middle">time / compute-only time (%)</text>'
    )
    legend = [
        ("wall", colors["wall"]),
        ("read total", colors["read"]),
        ("exposed read", colors["exposed"]),
        ("Relative expert throughput", colors["throughput"]),
    ]
    lx, ly = left + 12, top + 10
    for i, (name, color) in enumerate(legend):
        yy = ly + i * 20
        if name == "Relative expert throughput":
            parts.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{lx+11}" cy="{yy}" r="3" fill="{color}"/>')
        else:
            parts.append(f'<rect x="{lx}" y="{yy-8}" width="16" height="12" fill="{color}"/>')
        parts.append(f'<text class="small" x="{lx+28}" y="{yy+4}">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def maybe_plot(path: Path, rows: list[dict[str, object]]) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        svg_path = path.with_suffix(".svg")
        write_svg_plot(svg_path, rows)
        convert = shutil.which("convert")
        if convert:
            subprocess.run([convert, str(svg_path), str(path)], check=True)
            return f"{path} and {svg_path} (svg fallback: {exc})"
        return f"{svg_path} (svg fallback: {exc})"

    x = [int(r["hit_experts"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.bar(x, [float(r["read_exposed_pct_of_compute"]) for r in rows], label="exposed read")
    ax1.plot(x, [float(r["wall_pct_of_compute"]) for r in rows], marker="o", label="wall")
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        [
            float(
                r.get(
                    "relative_expert_throughput_pct",
                    r.get("relative_expert_compute_util_pct", r["relative_mfu_pct"]),
                )
            )
            for r in rows
        ],
        marker="s",
        color="black",
        label="Relative expert throughput",
    )
    ax1.set_xlabel("hit experts out of 10")
    ax1.set_ylabel("time / compute-only time (%)")
    ax2.set_ylabel("Rel. expert throughput (%)")
    ax1.grid(axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def run_styled_plot_script(summary_csv: Path, output_stem: Path) -> str | None:
    plot_script = Path(__file__).with_name("plot_qwen35_397b_int4_pipeline_read_compute.py")
    if not plot_script.exists():
        return None

    candidates: list[str] = []
    env_python = os.environ.get("PAGEDMOE_PLOT_PYTHON")
    if env_python:
        candidates.append(env_python)
    python_on_path = shutil.which("python")
    if python_on_path:
        candidates.append(python_on_path)
    candidates.append(sys.executable)

    seen: set[str] = set()
    for python in candidates:
        if python in seen:
            continue
        seen.add(python)
        try:
            subprocess.run(
                [
                    python,
                    str(plot_script),
                    "--summary-csv",
                    str(summary_csv),
                    "--output-stem",
                    str(output_stem),
                ],
                check=True,
            )
            return (
                f"{output_stem.with_suffix('.pdf')}, "
                f"{output_stem.with_suffix('.png')}, "
                f"{output_stem.with_suffix('.svg')}"
            )
        except Exception:
            continue
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-397B low-bit SSD read plus pure CPU PyTorch expert compute benchmark."
    )
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--pack-file", type=Path, default=None)
    parser.add_argument("--repack", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--layer-index", type=int, default=43)
    parser.add_argument("--expert-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--hit-experts", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    parser.add_argument("--precision-bits", type=int, default=8)
    parser.add_argument(
        "--compute-backend",
        choices=["bf16", "torchao-w8a16"],
        default="torchao-w8a16",
    )
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--baseline-repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--numa-node", type=int, default=1)
    parser.add_argument("--no-drop-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.expert_ids) != 10:
        raise SystemExit("--expert-ids must contain exactly 10 experts")
    if args.loader_workers < 1:
        raise SystemExit("--loader-workers must be >= 1")
    bound_cpus = bind_numa_node(args.numa_node)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.storage_root)
    pack_format: str
    if args.compute_backend == "torchao-w8a16":
        pack_file = args.pack_file or default_w8a16_pack_file(args.layer_index, args.expert_ids)
        print("loading 10 real Qwen3.5-397B experts and quantizing to torchao W8A16...")
        experts = load_w8a16_experts(args.model_root, args.layer_index, args.expert_ids)
        print("ensuring contiguous 10-expert W8A16 int8-weight pack...")
        pack_file, bytes_per_expert = ensure_w8a16_pack(
            pack_file=pack_file,
            layer_index=args.layer_index,
            expert_ids=args.expert_ids,
            experts=experts,
            repack=args.repack,
        )
        pack_format = "torchao_w8a16_int8_weight_payload"
        prefix = "qwen35_397b_w8a16_pipeline_read_compute"
        bytes_label = f"w8a16_bytes_per_expert={bytes_per_expert}"
    else:
        print("loading 10 real Qwen3.5-397B BF16 experts for pure CPU PyTorch compute...")
        experts = load_dense_experts(args.model_root, args.layer_index, args.expert_ids)
        pack_file = args.pack_file or default_bf16_pack_file(args.layer_index, args.expert_ids)
        print("ensuring contiguous 10-expert BF16 raw-weight pack...")
        pack_file, bytes_per_expert = ensure_bf16_pack(
            pack_file=pack_file,
            layer_index=args.layer_index,
            expert_ids=args.expert_ids,
            experts=experts,
            repack=args.repack,
        )
        pack_format = "bf16_raw_weight_payload"
        prefix = "qwen35_397b_bf16_pipeline_read_compute"
        bytes_label = f"bf16_bytes_per_expert={bytes_per_expert}"

    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"compute_backend={args.compute_backend}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"storage_root={args.storage_root}")
    print(f"compact_pack_file={pack_file}")
    print(f"compact_pack_format={pack_format}")
    print(f"compact_pack_bytes={pack_file.stat().st_size}")
    print("compact_pack_direct_io=True")
    print(f"model_root={args.model_root}")
    print(
        "shape="
        f"hidden:{manifest.hidden_size} intermediate:{manifest.intermediate_size} "
        f"experts_per_layer:{manifest.experts_per_layer} blocks:{manifest.num_blocks}"
    )
    print(bytes_label)
    print(f"layer_index={args.layer_index} expert_ids={args.expert_ids}")
    print(f"numa_node={args.numa_node} affinity_cpus={bound_cpus}")
    print(f"torch_threads={torch.get_num_threads()} loader_workers={args.loader_workers}")

    x = torch.randn(args.tokens, manifest.hidden_size, dtype=torch.bfloat16)
    assert_cpu_experts(experts, x)

    print("running compute-only baseline...")
    baseline_samples, checksum = run_compute_only(
        experts, x, args.expert_ids, args.baseline_repeats, args.warmup
    )
    baseline_compute_ms = median(baseline_samples)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    print(f"compute-only median for 10 experts = {baseline_compute_ms:.3f} ms (100%)")
    print("hit miss wall% compute% read% exposed_read% overlap% rel_throughput%")

    with CompactPackReader(
        pack_file,
        args.expert_ids,
        bytes_per_expert,
    ) as compact_reader, ThreadPoolExecutor(max_workers=args.loader_workers) as loader_pool:
        for future in [loader_pool.submit(lambda: None) for _ in range(args.loader_workers)]:
            future.result()
        for hit_count in args.hit_experts:
            hit_ids = args.expert_ids[:hit_count]
            miss_ids = args.expert_ids[hit_count:]
            per_repeat: list[dict[str, float | int | str]] = []
            for repeat in range(args.repeats):
                row = run_pipeline_once(
                    reader=compact_reader,
                    experts=experts,
                    hit_ids=hit_ids,
                    miss_ids=miss_ids,
                    x=x,
                    loader_pool=loader_pool,
                    drop_cache=not args.no_drop_cache,
                )
                row["hit_experts"] = hit_count
                row["miss_experts"] = len(miss_ids)
                row["repeat"] = repeat
                row["hit_ids"] = " ".join(str(v) for v in hit_ids)
                row["miss_ids"] = " ".join(str(v) for v in miss_ids)
                per_repeat.append(row)
                detail_rows.append(row)
                checksum += float(row["checksum"])

            def med(key: str) -> float:
                return median([float(r[key]) for r in per_repeat])

            summary = {
                "hit_experts": hit_count,
                "miss_experts": len(miss_ids),
                "compute_only_ms": baseline_compute_ms,
                "wall_ms": med("wall_ms"),
                "compute_sum_ms": med("compute_sum_ms"),
                "read_sum_ms": med("read_sum_ms"),
                "read_union_ms": med("read_union_ms"),
                "read_exposed_ms": med("read_exposed_ms"),
                "read_overlap_ms": med("read_overlap_ms"),
                "read_offset": int(median([float(r["read_offset"]) for r in per_repeat])),
                "read_length": int(median([float(r["read_length"]) for r in per_repeat])),
                "bytes_read": int(median([float(r["bytes_read"]) for r in per_repeat])),
            }
            relative_expert_throughput_pct = (
                100.0
                if len(miss_ids) == 0
                else pct(baseline_compute_ms, float(summary["wall_ms"]))
            )
            summary.update(
                {
                    "wall_pct_of_compute": pct(float(summary["wall_ms"]), baseline_compute_ms),
                    "compute_sum_pct_of_compute": pct(float(summary["compute_sum_ms"]), baseline_compute_ms),
                    "read_sum_pct_of_compute": pct(float(summary["read_sum_ms"]), baseline_compute_ms),
                    "read_union_pct_of_compute": pct(float(summary["read_union_ms"]), baseline_compute_ms),
                    "read_exposed_pct_of_compute": pct(float(summary["read_exposed_ms"]), baseline_compute_ms),
                    "read_overlap_pct_of_compute": pct(float(summary["read_overlap_ms"]), baseline_compute_ms),
                    "relative_expert_throughput_pct": relative_expert_throughput_pct,
                    "relative_expert_compute_util_pct": relative_expert_throughput_pct,
                    "relative_mfu_pct": relative_expert_throughput_pct,
                }
            )
            summary_rows.append(summary)
            print(
                f"{hit_count:>3} {len(miss_ids):>4} "
                f"{summary['wall_pct_of_compute']:>6.1f} "
                f"{summary['compute_sum_pct_of_compute']:>8.1f} "
                f"{summary['read_sum_pct_of_compute']:>6.1f} "
                f"{summary['read_exposed_pct_of_compute']:>13.1f} "
                f"{summary['read_overlap_pct_of_compute']:>8.1f} "
                f"{summary['relative_expert_throughput_pct']:>8.1f}"
            )

    detail_csv = args.output_dir / f"{prefix}_detail.csv"
    summary_csv = args.output_dir / f"{prefix}_summary.csv"
    summary_json = args.output_dir / f"{prefix}_summary.json"
    plot_path = args.output_dir / f"{prefix}.png"
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "config": {
                    "storage_root": str(args.storage_root),
                    "compact_pack_file": str(pack_file),
                    "compact_pack_format": pack_format,
                    "compact_pack_bytes": pack_file.stat().st_size,
                    "bytes_per_expert": bytes_per_expert,
                    "compact_pack_direct_io": True,
                    "model_root": str(args.model_root),
                    "compute_backend": args.compute_backend,
                    "layer_index": args.layer_index,
                    "expert_ids": args.expert_ids,
                    "precision_bits": args.precision_bits,
                    "tokens": args.tokens,
                    "repeats": args.repeats,
                    "baseline_repeats": args.baseline_repeats,
                    "warmup": args.warmup,
                    "loader_workers": args.loader_workers,
                    "torch_threads": args.torch_threads,
                    "numa_node": args.numa_node,
                    "drop_cache": not args.no_drop_cache,
                    "python": sys.executable,
                    "torch": torch.__version__,
                },
                "baseline_compute_ms_samples": baseline_samples,
                "summary": summary_rows,
                "checksum": checksum,
            },
            indent=2,
        )
        + "\n"
    )
    output_stem = plot_path.with_suffix("")
    plot_result = run_styled_plot_script(summary_csv, output_stem)
    if plot_result is None:
        plot_result = maybe_plot(plot_path, summary_rows)
    print(f"detail_csv={detail_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"summary_json={summary_json}")
    print(f"plot={plot_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

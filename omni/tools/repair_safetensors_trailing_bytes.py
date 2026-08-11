#!/usr/bin/env python3
"""Repair safetensors files with unreferenced trailing bytes.

Some otherwise valid safetensors files have an opaque footer appended after
the tensor data.  The safetensors format requires the header to cover the
entire file, so ``safe_open`` rejects those files with ``file not fully
covered``.  This tool validates the header and every tensor range, then writes
a new file containing exactly the bytes covered by the header.

The source file is never modified.  Run this with the same Python environment
as ComfyUI so the final ``safetensors.safe_open`` validation is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MAX_HEADER_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_TRAILING_BYTES = 1024 * 1024

# Current safetensors scalar types.  Sub-byte types use fractional bytes.
_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E8M0": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    "U4": 4,
    "I4": 4,
}


class RepairError(RuntimeError):
    """Raised when a file cannot be repaired without guessing."""


@dataclass(frozen=True)
class Inspection:
    path: str
    file_size: int
    header_size: int
    data_start: int
    tensor_count: int
    declared_data_size: int
    expected_file_size: int
    trailing_bytes: int
    source_sha256: str


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _tensor_nbytes(name: str, entry: dict[str, Any]) -> int:
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    if dtype not in _DTYPE_BITS:
        raise RepairError(f"{name}: unsupported dtype {dtype!r}")
    if not isinstance(shape, list) or any(not _is_int(dim) or dim < 0 for dim in shape):
        raise RepairError(f"{name}: invalid shape {shape!r}")

    elements = math.prod(shape)
    total_bits = elements * _DTYPE_BITS[dtype]
    if total_bits % 8:
        raise RepairError(f"{name}: tensor size is not byte aligned")
    return total_bits // 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_file(path: Path) -> Inspection:
    path = path.resolve(strict=True)
    file_size = path.stat().st_size
    if file_size < 8:
        raise RepairError("file is too short to contain a safetensors header")

    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size == 0 or header_size > MAX_HEADER_BYTES:
            raise RepairError(f"invalid header size: {header_size}")
        data_start = 8 + header_size
        if data_start > file_size:
            raise RepairError(
                f"truncated header: needs {data_start} bytes, file has {file_size}"
            )
        header_bytes = handle.read(header_size)

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"invalid header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise RepairError("safetensors header must be a JSON object")

    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict):
                raise RepairError("__metadata__ must be an object")
            continue
        if not isinstance(entry, dict):
            raise RepairError(f"{name}: tensor entry must be an object")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_is_int(value) for value in offsets)
        ):
            raise RepairError(f"{name}: invalid data_offsets {offsets!r}")
        start, end = offsets
        if start < 0 or end < start:
            raise RepairError(f"{name}: invalid data range [{start}, {end})")
        expected_nbytes = _tensor_nbytes(name, entry)
        if end - start != expected_nbytes:
            raise RepairError(
                f"{name}: range is {end - start} bytes, shape/dtype requires "
                f"{expected_nbytes}"
            )
        ranges.append((start, end, name))

    nonempty_ranges = sorted(item for item in ranges if item[0] != item[1])
    cursor = 0
    for start, end, name in nonempty_ranges:
        if start > cursor:
            raise RepairError(
                f"uncovered data gap [{cursor}, {start}) before tensor {name!r}"
            )
        if start < cursor:
            raise RepairError(
                f"overlapping tensor data at [{start}, {cursor}) for {name!r}"
            )
        cursor = end

    for start, end, name in ranges:
        if start == end and start > cursor:
            raise RepairError(
                f"empty tensor {name!r} points beyond declared data size {cursor}"
            )

    expected_file_size = data_start + cursor
    trailing_bytes = file_size - expected_file_size
    if trailing_bytes < 0:
        raise RepairError(
            f"truncated tensor data: needs {expected_file_size} bytes, "
            f"file has {file_size}"
        )

    return Inspection(
        path=str(path),
        file_size=file_size,
        header_size=header_size,
        data_start=data_start,
        tensor_count=len(ranges),
        declared_data_size=cursor,
        expected_file_size=expected_file_size,
        trailing_bytes=trailing_bytes,
        source_sha256=_sha256(path),
    )


def default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_fixed{source.suffix}")


def _validate_with_safetensors(path: Path, expected_tensor_count: int) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RepairError(
            "safetensors is not installed; run this script with ComfyUI's Python"
        ) from exc

    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor_count = len(list(handle.keys()))
    except Exception as exc:
        raise RepairError(f"repaired copy failed safetensors validation: {exc}") from exc
    if tensor_count != expected_tensor_count:
        raise RepairError(
            f"repaired copy exposes {tensor_count} tensors; expected "
            f"{expected_tensor_count}"
        )


def repair_file(
    source: Path,
    output: Path,
    *,
    max_trailing_bytes: int = DEFAULT_MAX_TRAILING_BYTES,
    expected_trailing_bytes: int | None = None,
    expected_sha256: str | None = None,
    force: bool = False,
) -> tuple[Inspection, str]:
    inspection = inspect_file(source)
    if expected_sha256 is not None:
        normalized = expected_sha256.strip().lower()
        if inspection.source_sha256 != normalized:
            raise RepairError(
                "source SHA256 mismatch: "
                f"expected {normalized}, got {inspection.source_sha256}"
            )
    if inspection.trailing_bytes == 0:
        raise RepairError("file is already fully covered; no repair is needed")
    if inspection.trailing_bytes > max_trailing_bytes:
        raise RepairError(
            f"refusing to remove {inspection.trailing_bytes} trailing bytes; "
            f"limit is {max_trailing_bytes}"
        )
    if (
        expected_trailing_bytes is not None
        and inspection.trailing_bytes != expected_trailing_bytes
    ):
        raise RepairError(
            f"expected {expected_trailing_bytes} trailing bytes, found "
            f"{inspection.trailing_bytes}"
        )

    source = source.resolve(strict=True)
    output = output.resolve(strict=False)
    if source == output:
        raise RepairError("output must not be the source file")
    if output.exists() and not force:
        raise RepairError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    output_digest = hashlib.sha256()
    try:
        remaining = inspection.expected_file_size
        with source.open("rb") as src, temporary.open("xb") as dst:
            while remaining:
                chunk = src.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise RepairError("source ended while creating repaired copy")
                dst.write(chunk)
                output_digest.update(chunk)
                remaining -= len(chunk)
            dst.flush()
            os.fsync(dst.fileno())

        _validate_with_safetensors(temporary, inspection.tensor_count)
        if output.exists():
            output.unlink()
        os.replace(temporary, output)
        shutil.copystat(source, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return inspection, output_digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create validated copies of safetensors files whose metadata does "
            "not cover an appended footer. Source files are never modified."
        )
    )
    parser.add_argument("sources", nargs="+", type=Path, help="input .safetensors files")
    parser.add_argument(
        "--output",
        type=Path,
        help="output path (valid only with one input; default: *_fixed.safetensors)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="inspect files and print reports without creating repaired copies",
    )
    parser.add_argument(
        "--expected-sha256",
        help="require this source SHA256 (valid only with one input)",
    )
    parser.add_argument(
        "--expected-trailing-bytes",
        type=int,
        help="require an exact footer size, for example 64",
    )
    parser.add_argument(
        "--max-trailing-bytes",
        type=int,
        default=DEFAULT_MAX_TRAILING_BYTES,
        help=f"maximum removable footer size (default: {DEFAULT_MAX_TRAILING_BYTES})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output file; never replaces the source",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON reports")
    return parser


def _print_report(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(f"source: {report['path']}")
    print(f"  SHA256: {report['source_sha256']}")
    print(f"  tensors: {report['tensor_count']}")
    print(f"  actual size: {report['file_size']}")
    print(f"  declared size: {report['expected_file_size']}")
    print(f"  trailing bytes: {report['trailing_bytes']}")
    if "output" in report:
        print(f"  repaired copy: {report['output']}")
        print(f"  repaired SHA256: {report['output_sha256']}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output is not None and len(args.sources) != 1:
        print("error: --output requires exactly one source", file=sys.stderr)
        return 2
    if args.expected_sha256 is not None and len(args.sources) != 1:
        print("error: --expected-sha256 requires exactly one source", file=sys.stderr)
        return 2
    if args.max_trailing_bytes < 0:
        print("error: --max-trailing-bytes must be non-negative", file=sys.stderr)
        return 2
    if args.expected_trailing_bytes is not None and args.expected_trailing_bytes < 0:
        print("error: --expected-trailing-bytes must be non-negative", file=sys.stderr)
        return 2

    failed = False
    for source in args.sources:
        try:
            if args.check_only:
                inspection = inspect_file(source)
                report = asdict(inspection)
            else:
                output = args.output or default_output_path(source)
                inspection, output_sha256 = repair_file(
                    source,
                    output,
                    max_trailing_bytes=args.max_trailing_bytes,
                    expected_trailing_bytes=args.expected_trailing_bytes,
                    expected_sha256=args.expected_sha256,
                    force=args.force,
                )
                report = asdict(inspection)
                report.update(
                    output=str(output.resolve()),
                    output_sha256=output_sha256,
                )
            _print_report(report, args.json)
        except (OSError, RepairError) as exc:
            failed = True
            print(f"error: {source}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

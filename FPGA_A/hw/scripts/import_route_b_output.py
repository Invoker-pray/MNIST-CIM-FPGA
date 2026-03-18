#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

WEIGHT_FILES = {
    "fc1_weight_int8.hex",
    "fc1_bias_int32.hex",
    "fc2_weight_int8.hex",
    "fc2_bias_int32.hex",
}

QUANT_FILES = {
    "quant_params.hex",
    "quant_config.json",
}

SAMPLE_FILES_EXACT = {
    "mnist_samples_route_b_output_2.hex",
}

EXPECTED_FILES_EXACT = {
    "labels.txt",
    "preds.txt",
}

SAMPLE_PREFIXES = ("input_",)
EXPECTED_PREFIXES = ("pred_",)


def classify_file(name: str) -> str | None:
    if name in WEIGHT_FILES:
        return "weights"
    if name in QUANT_FILES:
        return "quant"
    if name in SAMPLE_FILES_EXACT or name.startswith(SAMPLE_PREFIXES):
        return "samples"
    if name in EXPECTED_FILES_EXACT or name.startswith(EXPECTED_PREFIXES):
        return "expected"
    return None


def collect_source_files(src_dir: Path) -> list[Path]:
    # 只收集一层和标准子目录里的文件，避免误扫太多无关内容
    candidates: list[Path] = []

    for p in src_dir.iterdir():
        if p.is_file():
            candidates.append(p)

    for sub in ("weights", "quant", "samples", "expected"):
        subdir = src_dir / sub
        if subdir.is_dir():
            for p in subdir.iterdir():
                if p.is_file():
                    candidates.append(p)

    # 去重
    uniq = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import files from route_b_output into current project's data/ tree."
    )
    parser.add_argument(
        "src_dir",
        type=Path,
        help="Source directory, e.g. route_b_output",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=Path("data"),
        help="Destination data directory (default: ./data)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned operations, do not modify files",
    )
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    dst_dir = args.dst_dir.resolve()

    if not src_dir.is_dir():
        raise SystemExit(f"ERROR: source directory does not exist: {src_dir}")

    for sub in ("weights", "quant", "samples", "expected"):
        (dst_dir / sub).mkdir(parents=True, exist_ok=True)

    files = collect_source_files(src_dir)
    if not files:
        raise SystemExit(f"ERROR: no files found in source directory: {src_dir}")

    copied = 0
    skipped = 0

    for src in files:
        category = classify_file(src.name)
        if category is None:
            print(f"SKIP: unrecognized file -> {src.name}")
            skipped += 1
            continue

        dst = dst_dir / category / src.name
        op = "MOVE" if args.move else "COPY"
        print(f"{op}: {src} -> {dst}")

        if not args.dry_run:
            if args.move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(src, dst)

        copied += 1

    print("--------------------------------------------------")
    print(f"Done. handled={copied}, skipped={skipped}")
    print(f"Destination root: {dst_dir}")


if __name__ == "__main__":
    main()

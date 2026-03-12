#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
from typing import List


def read_hex_lines(path: pathlib.Path) -> List[int]:
    vals: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            try:
                vals.append(int(s, 16))
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: invalid hex entry: {s}") from e
    return vals


def write_packed_lines(
    path: pathlib.Path,
    words: List[int],
    word_bits: int,
    line_elems: int,
) -> None:
    """
    Pack `line_elems` words into one hex line.

    IMPORTANT:
      This packing matches the current SystemVerilog unpacking style:

          packed[elem_idx*word_bits +: word_bits]

      Therefore:
        - element 0 goes to the LEAST-significant bits
        - element 1 goes to the next word_bits
        - ...
        - when printed as hex, element 0 appears at the RIGHTMOST side
    """
    if line_elems <= 0:
        raise ValueError("line_elems must be > 0")
    if word_bits <= 0:
        raise ValueError("word_bits must be > 0")

    if len(words) % line_elems != 0:
        raise ValueError(
            f"{path.name}: number of words {len(words)} is not divisible by line_elems {line_elems}"
        )

    mask = (1 << word_bits) - 1
    hex_width = (word_bits * line_elems + 3) // 4

    with path.open("w", encoding="utf-8") as f:
        for base in range(0, len(words), line_elems):
            packed = 0
            for i in range(line_elems):
                packed |= (words[base + i] & mask) << (i * word_bits)
            f.write(f"{packed:0{hex_width}x}\n")


def convert_fc1_weight(
    src: pathlib.Path,
    dst: pathlib.Path,
    out_tile: int,
    in_tile: int,
    weight_bits: int,
    total_out: int = 16,
    total_in: int = 784,
) -> None:
    words = read_hex_lines(src)
    # words[out * total_in + in] = W[out][in]  (row-major)

    # RTL 期望: tile[ob*N_IB+ib][tr*in_tile+tc] = W[ob*out_tile+tr][ib*in_tile+tc]
    n_ob = total_out // out_tile  # = 1
    n_ib = total_in // in_tile  # = 49

    reordered = []
    for ob in range(n_ob):
        for ib in range(n_ib):
            for tr in range(out_tile):  # output neuron offset
                for tc in range(in_tile):  # input pixel offset
                    out_idx = ob * out_tile + tr
                    in_idx = ib * in_tile + tc
                    reordered.append(words[out_idx * total_in + in_idx])

    write_packed_lines(dst, reordered, weight_bits, out_tile * in_tile)


def convert_fc1_bias(
    src: pathlib.Path,
    dst: pathlib.Path,
    tile_out: int,
    bias_bits: int,
) -> None:
    words = read_hex_lines(src)
    write_packed_lines(dst, words, bias_bits, tile_out)


def convert_sample_tiles(
    src: pathlib.Path,
    dst: pathlib.Path,
    tile_in: int,
    input_bits: int,
) -> None:
    words = read_hex_lines(src)
    write_packed_lines(dst, words, input_bits, tile_in)


def convert_tree(
    data_dir: pathlib.Path,
    out_dir: pathlib.Path,
    tile_in: int = 16,
    tile_out: int = 16,
    input_bits: int = 8,
    weight_bits: int = 8,
    bias_bits: int = 32,
) -> None:
    weights_in = data_dir / "weights"
    samples_in = data_dir / "samples"
    expected_in = data_dir / "expected"
    quant_in = data_dir / "quant"

    weights_out = out_dir / "weights"
    samples_out = out_dir / "samples"
    expected_out = out_dir / "expected"
    quant_out = out_dir / "quant"

    for d in (weights_out, samples_out, expected_out, quant_out):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Files whose format changes
    # ------------------------------------------------------------
    convert_fc1_weight(
        weights_in / "fc1_weight_int8.hex",
        weights_out / "fc1_weight_int8.hex",
        tile_out,
        tile_in,
        weight_bits,
    )

    convert_fc1_bias(
        weights_in / "fc1_bias_int32.hex",
        weights_out / "fc1_bias_int32.hex",
        tile_out,
        bias_bits,
    )

    convert_sample_tiles(
        samples_in / "mnist_samples_route_b_output_2.hex",
        samples_out / "mnist_samples_route_b_output_2.hex",
        tile_in,
        input_bits,
    )

    for src in sorted(samples_in.glob("input_*.hex")):
        convert_sample_tiles(src, samples_out / src.name, tile_in, input_bits)

    # ------------------------------------------------------------
    # Files whose format stays unchanged
    # ------------------------------------------------------------
    passthrough = [
        weights_in / "fc2_weight_int8.hex",
        weights_in / "fc2_bias_int32.hex",
        quant_in / "quant_params.hex",
        quant_in / "quant_config.json",
    ]

    for src in passthrough:
        dst = out_dir / src.relative_to(data_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    for src in expected_in.glob("*"):
        dst = expected_out / src.name
        dst.write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert old element-per-line FPGA data hex files into packed tile/block format."
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help="Input data directory containing weights/, samples/, quant/, expected/.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="Output directory to write converted data tree.",
    )
    parser.add_argument("--tile-in", type=int, default=16)
    parser.add_argument("--tile-out", type=int, default=16)
    parser.add_argument("--input-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--bias-bits", type=int, default=32)

    args = parser.parse_args()

    convert_tree(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        tile_in=args.tile_in,
        tile_out=args.tile_out,
        input_bits=args.input_bits,
        weight_bits=args.weight_bits,
        bias_bits=args.bias_bits,
    )

    print(f"Converted data tree written to: {args.out_dir}")
    print(
        "Reminder: fc1/sample/input hex files are now packed tile/block format, not element-per-line format."
    )


if __name__ == "__main__":
    main()

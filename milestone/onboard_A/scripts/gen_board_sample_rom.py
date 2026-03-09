from pathlib import Path
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: python gen_board_sample_rom.py <route_b_output_dir> <out_hex>")
        sys.exit(1)

    src_dir = Path(sys.argv[1])
    out_hex = Path(sys.argv[2])

    all_lines = []
    for i in range(20):
        f = src_dir / f"input_{i}.hex"
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}")

        lines = [x.strip() for x in f.read_text().splitlines() if x.strip()]
        if len(lines) != 784:
            raise ValueError(f"{f} has {len(lines)} lines, expected 784")
        all_lines.extend(lines)

    out_hex.parent.mkdir(parents=True, exist_ok=True)
    out_hex.write_text("\n".join(all_lines) + "\n")
    print(f"Wrote {out_hex} with {len(all_lines)} lines")


if __name__ == "__main__":
    main()

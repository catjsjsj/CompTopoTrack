from __future__ import annotations

import argparse
from pathlib import Path

from _sim_plus import build_basic_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/basic_features"))
    parser.add_argument("--sequences", nargs="+", default=["01", "02"])
    parser.add_argument("--segmentation-suffix", default="_GT/TRA")
    args = parser.parse_args()

    output = build_basic_features(
        args.data_root.resolve(),
        args.output.resolve(),
        args.sequences,
        args.segmentation_suffix,
    )
    print(output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from _sim_plus import (
    ROOT,
    add_object_features,
    ensure_csv_alias,
    extract_metric_features,
    python,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", default="02")
    parser.add_argument("--segmentation-suffix", default="_GT/TRA")
    parser.add_argument(
        "--metric-weights",
        type=Path,
        default=ROOT / "weights/metric_encoder_all_params.pth",
    )
    parser.add_argument(
        "--gnn-checkpoint",
        type=Path,
        default=ROOT / "weights/association_gnn/checkpoints/best.ckpt",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--output", type=Path, default=Path("outputs/02_RES"))
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--k-sister", type=float, default=2.0)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    work_dir = args.work_dir.resolve()
    output = args.output.resolve()
    features = extract_metric_features(
        data_root,
        work_dir / "features",
        args.metric_weights.resolve(),
        [args.sequence],
        args.segmentation_suffix,
        args.image_batch_size,
    )
    add_object_features(data_root, features, args.sequence, args.segmentation_suffix)
    ensure_csv_alias(features, args.sequence)

    run(
        [
            python(),
            str(ROOT / "src/inference/inference_clean.py"),
            "-mp",
            str(args.gnn_checkpoint.resolve()),
            "-ns",
            args.sequence,
            "-oc",
            str(features),
        ]
    )
    inference_dir = features / f"{args.sequence}_RES_inference"
    segmentation_dir = data_root / f"{args.sequence}{args.segmentation_suffix}"
    run(
        [
            python(),
            str(ROOT / "src/inference/postprocess_clean.py"),
            "-modality",
            "2D",
            "-iseg",
            str(segmentation_dir),
            "-oi",
            str(inference_dir),
            "-output_dir",
            str(output),
            "-decision_threshold",
            str(args.decision_threshold),
            "-K_sister",
            str(args.k_sister),
            "-division_mode",
            "geometric",
        ]
    )
    print(output)


if __name__ == "__main__":
    main()

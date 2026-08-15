from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _sim_plus import ROOT, build_basic_features, feature_root


sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--basic-output", type=Path, default=Path("outputs/basic_features"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metric"))
    parser.add_argument("--source-sequence", default="01")
    parser.add_argument("--train-frames", type=int, default=45)
    parser.add_argument("--validation-frames", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--roi-scale", type=float, default=1.0)
    parser.add_argument("--enabled-layers", nargs="+", type=int, default=[1])
    parser.add_argument("--gamma-direct", type=float, default=0.1)
    parser.add_argument("--gamma-closure", type=float, default=0.1)
    parser.add_argument("--degree-weight", type=float, default=0.0)
    parser.add_argument("--same-frame-weight", type=float, default=0.0)
    parser.add_argument("--closure-max-component-size", type=int, default=50)
    parser.add_argument("--segmentation-suffix", default="_GT/TRA")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    basic_output = args.basic_output.resolve()
    basic_root = feature_root(basic_output, data_root)
    if not (basic_root / args.source_sequence / "csv").is_dir():
        build_basic_features(
            data_root,
            basic_output,
            [args.source_sequence],
            args.segmentation_suffix,
        )

    import run_train_topology_metric_learning as trainer

    trainer.CONFIG.update(
        {
            "seed": args.seed,
            "num_epochs": args.epochs,
            "early_stop_patience": args.patience,
            "val_interval": 1,
            "effective_batch_size": 2,
            "enabled_layers": sorted(set(args.enabled_layers)),
            "same_frame_cluster_contrast": args.same_frame_weight > 0,
            "same_frame_cluster_weight": args.same_frame_weight,
            "gamma_direct": args.gamma_direct,
            "gamma_closure": args.gamma_closure,
            "degree_weight": args.degree_weight,
            "alpha_max": args.degree_weight,
            "closure_max_component_size": args.closure_max_component_size,
            "roi_mul_vals": (args.roi_scale, args.roi_scale),
            "data_dir_img": str(data_root),
            "data_dir_mask": str(data_root),
            "subdir_mask": args.segmentation_suffix.lstrip("_"),
            "dir_csv": str(basic_root),
            "num_sequences": 1,
            "sequences_names": [args.source_sequence],
            "start_index": int(args.source_sequence),
            "train_val_test_split": [args.train_frames, args.validation_frames, 0],
            "deviation": "with_overlap",
            "exp_name": "sim_plus_metric",
            "output_root": str(args.output.resolve()),
        }
    )
    summary = trainer.main()
    print(json.dumps({"all_params": summary["all_params"], "best_mrr": summary["best_mrr"]}, indent=2))


if __name__ == "__main__":
    main()

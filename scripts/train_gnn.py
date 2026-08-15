from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _sim_plus import (
    DROP_FEATURES,
    ROOT,
    add_object_features,
    extract_metric_features,
    link_csv_range,
    python,
    run,
    training_bbox_scale,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--metric-weights",
        type=Path,
        default=ROOT / "weights/metric_encoder_all_params.pth",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/gnn_training"))
    parser.add_argument("--source-sequence", default="01")
    parser.add_argument("--train-frames", type=int, default=45)
    parser.add_argument("--validation-frames", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--segmentation-suffix", default="_GT/TRA")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_output = output / "features"
    features = extract_metric_features(
        data_root,
        feature_output,
        args.metric_weights.resolve(),
        [args.source_sequence],
        args.segmentation_suffix,
        args.image_batch_size,
    )
    add_object_features(
        data_root, features, args.source_sequence, args.segmentation_suffix
    )

    split_root = output / "splits"
    if split_root.exists():
        raise FileExistsError(f"Refusing to overwrite {split_root}")
    train_end = args.train_frames - 1
    validation_start = args.train_frames
    validation_end = validation_start + args.validation_frames - 1
    link_csv_range(
        features / args.source_sequence,
        split_root / "train",
        0,
        train_end,
    )
    link_csv_range(
        features / args.source_sequence,
        split_root / "valid",
        validation_start,
        validation_end,
    )
    shutil.copytree(split_root / "valid", split_root / "test", symlinks=True)
    motion_scale = training_bbox_scale(split_root / "train/csv")
    (output / "training_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "source_sequence": args.source_sequence,
                "train_frames": [0, train_end],
                "validation_frames": [validation_start, validation_end],
                "motion_scale": motion_scale,
                "message_passing_layers": 10,
                "batch_size": args.batch_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    gnn_run = output / "gnn_run"
    if gnn_run.exists():
        raise FileExistsError(f"Refusing to overwrite {gnn_run}")
    command = [
        python(),
        str(ROOT / "run.py"),
        "model=association",
        "datamodule=sim_plus",
        f"seed={args.seed}",
        f"trainer.max_epochs={args.epochs}",
        f"trainer.gpus={0 if args.cpu else 1}",
        f"callbacks.early_stopping.patience={args.patience}",
        f"datamodule.batch_size={args.batch_size}",
        f"datamodule.dataset_params.main_path={split_root}",
        "datamodule.dataset_params.dirs_path.train=[train]",
        "datamodule.dataset_params.dirs_path.valid=[valid]",
        "datamodule.dataset_params.dirs_path.test=[test]",
        "datamodule.dataset_params.exp_name=sim_plus_release_2D",
        "datamodule.dataset_params.mul_vals=[1.0,1.0,1.0]",
        "datamodule.dataset_params.filter_edges=true",
        "datamodule.dataset_params.normalize=false",
        "datamodule.dataset_params.drop_feat=[" + ",".join(DROP_FEATURES) + "]",
        "datamodule.dataset_params.motion_scale="
        f"{{row:{motion_scale['row']:.8g},col:{motion_scale['col']:.8g}}}",
        "model.model_params.kwargs.message_passing.kwargs.num_layers=10",
        f"hydra.run.dir={gnn_run}",
    ]
    run(command)
    checkpoints = sorted((gnn_run / "checkpoints").glob("epoch=*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one validation-best checkpoint, found {checkpoints}")
    print(checkpoints[0])


if __name__ == "__main__":
    main()

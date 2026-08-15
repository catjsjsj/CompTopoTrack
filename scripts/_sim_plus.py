from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DROP_FEATURES = [
    "frame_num",
    "area",
    "min_row_bb",
    "min_col_bb",
    "max_row_bb",
    "max_col_bb",
    "centroid_row",
    "centroid_col",
    "major_axis_length",
    "minor_axis_length",
    "max_intensity",
    "mean_intensity",
    "min_intensity",
]


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.(?:tif|tiff|csv)$)", path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse a frame number from {path}")
    return int(match.group(1))


def sequence_paths(data_root: Path, sequence: str, segmentation_suffix: str):
    image_dir = data_root / sequence
    segmentation_dir = data_root / f"{sequence}{segmentation_suffix}"
    images = sorted(image_dir.glob("*.tif*"), key=frame_number)
    masks = sorted(segmentation_dir.glob("*.tif*"), key=frame_number)
    if not images:
        raise FileNotFoundError(f"No TIFF images found in {image_dir}")
    if not masks:
        raise FileNotFoundError(f"No TIFF masks found in {segmentation_dir}")
    if len(images) != len(masks):
        raise ValueError(
            f"Image/mask frame count mismatch for {sequence}: "
            f"{len(images)} != {len(masks)}"
        )
    if [frame_number(path) for path in images] != [frame_number(path) for path in masks]:
        raise ValueError(f"Image/mask frame indices differ for {sequence}")
    return image_dir, segmentation_dir, images, masks


def feature_root(output_root: Path, data_root: Path) -> Path:
    return output_root / data_root.name


def build_basic_features(
    data_root: Path,
    output_root: Path,
    sequences: list[str],
    segmentation_suffix: str,
) -> Path:
    from src.datamodules.extract_features.preprocess_seq2graph_2d import create_csv

    for sequence in sequences:
        sequence_paths(data_root, sequence, segmentation_suffix)
    create_csv(
        input_images=str(data_root),
        input_masks=str(data_root),
        input_seg=str(data_root),
        input_model="",
        output_csv=str(output_root),
        basic=True,
        sequences=sequences,
        seg_dir=segmentation_suffix,
    )
    return feature_root(output_root, data_root)


def extract_metric_features(
    data_root: Path,
    output_root: Path,
    metric_weights: Path,
    sequences: list[str],
    segmentation_suffix: str,
    image_batch_size: int,
) -> Path:
    from src.datamodules.extract_features.preprocess_seq2graph_2d import create_csv

    for sequence in sequences:
        sequence_paths(data_root, sequence, segmentation_suffix)
    create_csv(
        input_images=str(data_root),
        input_masks=str(data_root),
        input_seg=str(data_root),
        input_model=str(metric_weights),
        output_csv=str(output_root),
        basic=False,
        sequences=sequences,
        seg_dir=segmentation_suffix,
        image_batch_size=image_batch_size,
    )
    return feature_root(output_root, data_root)


def add_object_features(
    data_root: Path,
    features: Path,
    sequence: str,
    segmentation_suffix: str,
) -> None:
    from src.datamodules.datasets.trackastra_object_features import (
        TRACKASTRA_OBJECT_COLUMNS,
        extract_trackastra_object_features,
    )

    _, segmentation_dir, images, masks = sequence_paths(
        data_root, sequence, segmentation_suffix
    )
    image_by_frame = {frame_number(path): path for path in images}
    mask_by_frame = {frame_number(path): path for path in masks}
    csv_dir = features / sequence / "csv"
    csv_files = sorted(csv_dir.glob("*.csv"), key=frame_number)
    if not csv_files:
        raise FileNotFoundError(f"No feature CSVs found in {csv_dir}")

    for csv_path in csv_files:
        frame = frame_number(csv_path)
        image = imread(image_by_frame[frame])
        mask = imread(mask_by_frame[frame])
        descriptors = extract_trackastra_object_features(mask, image)
        table = pd.read_csv(csv_path)
        label_column = "seg_label" if "seg_label" in table else "id"
        rows = []
        for label in table[label_column].astype(int):
            if label not in descriptors:
                raise ValueError(
                    f"Label {label} in {csv_path} is absent from {segmentation_dir}"
                )
            rows.append(descriptors[label])
        values = np.stack(rows) if rows else np.empty((0, len(TRACKASTRA_OBJECT_COLUMNS)))
        table.loc[:, list(TRACKASTRA_OBJECT_COLUMNS)] = values
        table.to_csv(csv_path, index=False)


def ensure_csv_alias(features: Path, sequence: str) -> Path:
    source = features / sequence
    alias = features / f"{sequence}_CSV"
    if not (source / "csv").is_dir():
        raise FileNotFoundError(source / "csv")
    if alias.is_symlink():
        if alias.resolve() == source.resolve():
            return alias
        alias.unlink()
    elif alias.exists():
        if (alias / "csv").is_dir():
            return alias
        raise FileExistsError(alias)
    alias.symlink_to(sequence, target_is_directory=True)
    return alias


def link_csv_range(source: Path, destination: Path, start: int, end: int) -> int:
    csv_dir = destination / "csv"
    csv_dir.mkdir(parents=True, exist_ok=False)
    selected = [
        path
        for path in sorted((source / "csv").glob("*.csv"), key=frame_number)
        if start <= frame_number(path) <= end
    ]
    if not selected:
        raise ValueError(f"No CSV frames selected in [{start}, {end}]")
    for path in selected:
        (csv_dir / path.name).symlink_to(path.resolve())
    return len(selected)


def training_bbox_scale(csv_dir: Path) -> dict[str, float]:
    required = {"min_row_bb", "max_row_bb", "min_col_bb", "max_col_bb"}
    row_scale = 0.0
    col_scale = 0.0
    files = sorted(csv_dir.glob("*.csv"), key=frame_number)
    if not files:
        raise FileNotFoundError(csv_dir)
    for path in files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for row in reader:
                row_scale = max(
                    row_scale,
                    abs(float(row["max_row_bb"]) - float(row["min_row_bb"])),
                )
                col_scale = max(
                    col_scale,
                    abs(float(row["max_col_bb"]) - float(row["min_col_bb"])),
                )
    if row_scale <= 0 or col_scale <= 0:
        raise ValueError("Invalid training bounding-box scale")
    return {"row": row_scale, "col": col_scale}


def run(command: list[str], cwd: Path = ROOT) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, cwd=str(cwd), env=environment, check=True)


def python() -> str:
    return sys.executable

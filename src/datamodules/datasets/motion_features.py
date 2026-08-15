

import numpy as np


def _positive_scale(value):
    value = float(value)
    return value if np.isfinite(value) and value > 0 else 1.0


def relative_motion_features(
    df_data,
    edge_index,
    roi,
    *,
    is_3d=False,
    jump_frames=1,
):
    
    coord_cols = ["centroid_row", "centroid_col"]
    scales = [_positive_scale(roi.get("row", 1.0)), _positive_scale(roi.get("col", 1.0))]
    if is_3d:
        coord_cols = ["centroid_depth", *coord_cols]
        scales = [_positive_scale(roi.get("depth", 1.0)), *scales]

    required = ["frame_num", *coord_cols]
    missing = [column for column in required if column not in df_data.columns]
    if missing:
        raise ValueError(f"Cannot build motion features; missing columns: {missing}")

    if hasattr(edge_index, "detach"):
        edges = edge_index.detach().cpu().numpy()
    else:
        edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edges.shape}")

    source, target = edges.astype(np.int64, copy=False)
    frame = df_data["frame_num"].to_numpy(dtype=np.float32)
    coords = df_data[coord_cols].to_numpy(dtype=np.float32)
    scale = np.asarray(scales, dtype=np.float32)

    temporal_scale = _positive_scale(jump_frames)
    dt = ((frame[target] - frame[source]) / temporal_scale)[:, None]
    displacement = (coords[target] - coords[source]) / scale[None, :]
    distance = np.linalg.norm(displacement, axis=1, keepdims=True)
    features = np.concatenate((dt, displacement, distance), axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Relative motion features contain non-finite values")
    return features.astype(np.float32, copy=False)

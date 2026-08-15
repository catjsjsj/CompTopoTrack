

from __future__ import annotations

import numpy as np
from skimage.measure import regionprops


TRACKASTRA_OBJECT_COLUMNS = (
    "trackastra_equivalent_diameter_area",
    "trackastra_intensity_mean",
    "trackastra_inertia_00",
    "trackastra_inertia_01",
    "trackastra_inertia_10",
    "trackastra_inertia_11",
    "trackastra_border_dist",
)


def _border_distance_image(shape: tuple[int, int], cutoff: int = 5) -> np.ndarray:
    
    border = np.ones(shape, dtype=np.float32)
    for axis, size in enumerate(shape):
        values = np.arange(cutoff, dtype=np.float32) / cutoff
        values = values[:size]

        low = [slice(None)] * 2
        low[axis] = slice(0, cutoff)
        low_values = values[(...,) + (None,) * (1 - axis)]
        border[tuple(low)] = np.minimum(border[tuple(low)], low_values)

        high = [slice(None)] * 2
        high[axis] = slice(max(0, size - cutoff), size)
        high_values = values[::-1][(...,) + (None,) * (1 - axis)]
        border[tuple(high)] = np.minimum(border[tuple(high)], high_values)
    return 1.0 - border


def extract_trackastra_object_features(
    mask: np.ndarray,
    image: np.ndarray,
) -> dict[int, np.ndarray]:
    
    mask = np.asarray(mask)
    image = np.asarray(image)
    if mask.ndim != 2 or image.ndim != 2:
        raise ValueError(
            f"Expected 2D mask and image, got {mask.shape} and {image.shape}"
        )
    if mask.shape != image.shape:
        raise ValueError(f"Mask/image shape mismatch: {mask.shape} != {image.shape}")

    border_image = _border_distance_image(mask.shape)
    border_by_label = {
        int(prop.label): float(prop.max_intensity)
        for prop in regionprops(mask, intensity_image=border_image)
    }
    result = {}
    for prop in regionprops(mask, intensity_image=image):
        inertia = np.asarray(prop.inertia_tensor, dtype=np.float32).reshape(2, 2)
        equivalent_diameter = getattr(
            prop, "equivalent_diameter_area", prop.equivalent_diameter
        )
        values = np.asarray(
            [
                equivalent_diameter,
                prop.mean_intensity,
                inertia[0, 0],
                inertia[0, 1],
                inertia[1, 0],
                inertia[1, 1],
                border_by_label[int(prop.label)],
            ],
            dtype=np.float32,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite Trackastra features for label {prop.label}")
        result[int(prop.label)] = values
    return result

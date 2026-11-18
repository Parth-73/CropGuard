"""Lossless predicted-mask encoding and original-image pixel coordinates."""

import numpy as np


def encode_rle(mask: np.ndarray) -> dict:
    """Row-major RLE; counts begin with the number of zero pixels (possibly zero)."""
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="C")
    if flat.size == 0:
        raise ValueError("Cannot encode an empty mask")
    if not np.isin(flat, [0, 1]).all():
        raise ValueError("Expected a binary mask")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, edges, flat.size]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "order": "C", "counts": counts}


def mask_geometry(mask: np.ndarray) -> dict:
    """Bounds use exclusive x2/y2; pixel centers are (column + .5, row + .5)."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Cannot locate an empty mask")
    center = [float(cols.mean() + 0.5), float(rows.mean() + 0.5)]
    # A centroid can fall outside a concave plant mask. Give an additional in-mask point.
    nearest = int(np.argmin((cols + 0.5 - center[0]) ** 2 + (rows + 0.5 - center[1]) ** 2))
    return {
        "mask_bbox_xyxy": [int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)],
        "centroid_xy": center,
        "mask_point_xy": [float(cols[nearest] + 0.5), float(rows[nearest] + 0.5)],
        "area_px": int(len(rows)),
    }

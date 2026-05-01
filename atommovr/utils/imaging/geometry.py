"""Geometry helpers shared across imaging utilities.

All functions operate on centroid coordinates stored as (row, column) or
(y, x) pairs indexed in image space (origin top-left). Positive rotation
angles follow OpenCV's convention: counter-clockwise rotation about the
image centre.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

# Avoid hard dependency on OpenCV at import time; use it when available.
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

CoordArray = np.ndarray


def _as_yx_array(points: Iterable[Tuple[float, float]] | np.ndarray) -> CoordArray:
    """Return points as an (N, 2) float array in (y, x) order."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape(-1, 2)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError("points array must have an even number of entries")
        arr = arr.reshape(-1, 2)
    if arr.shape[1] != 2:
        raise ValueError("points must be shaped (N, 2)")
    return arr


def rotate_points_ccw(
    points: Iterable[Tuple[float, float]] | np.ndarray,
    image_shape: Tuple[int, int],
    angle_deg: float,
) -> CoordArray:
    """Rotate points counter-clockwise by ``angle_deg`` degrees.

    Parameters
    ----------
    points:
        Iterable of (row, column) coordinates.
    image_shape:
        Image height and width; determines the rotation centre.
    angle_deg:
        Positive values rotate counter-clockwise, matching ``cv2``'s
        ``getRotationMatrix2D`` convention.
    """
    arr = _as_yx_array(points)
    if arr.size == 0:
        return arr
    h, w = image_shape[:2]
    center = (w / 2.0, h / 2.0)
    # Use a numpy-only rotation matrix to avoid importing OpenCV when not needed.
    # Prefer OpenCV's rotation matrix when available to match image rotation
    # semantics (avoids off-by-0.5 pixel differences that can change binning).
    if cv2 is not None:
        pts_xy1 = np.hstack([arr[:, ::-1], np.ones((arr.shape[0], 1))])
        M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
        transformed_xy = (M @ pts_xy1.T).T
        return transformed_xy[:, ::-1]

    theta = np.deg2rad(float(angle_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Center-based rotation: translate points to origin, rotate, translate back.
    # Points are in (y, x) order; convert to (x, y) for matrix ops.
    pts_xy = arr[:, ::-1]
    # shift to center
    pts_xy_centered = pts_xy - np.array([[center[0], center[1]]])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = (pts_xy_centered @ R.T) + np.array([[center[0], center[1]]])
    # convert back to (y, x)
    return rotated[:, ::-1]


def rotate_points_cw(
    points: Iterable[Tuple[float, float]] | np.ndarray,
    image_shape: Tuple[int, int],
    angle_deg: float,
) -> CoordArray:
    """Rotate points clockwise by ``angle_deg`` degrees."""
    return rotate_points_ccw(points, image_shape, -float(angle_deg))


__all__ = ["rotate_points_ccw", "rotate_points_cw"]

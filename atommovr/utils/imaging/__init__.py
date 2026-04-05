"""
Imaging utilities for atom array experiments.

This subpackage provides:
- Image synthesis of tweezer arrays with Gaussian PSFs
- Blob-based centroid extraction and subpixel refinement
- Multiple grid angle estimation techniques
- Grid fitting from centroids
- Lightweight visualization helpers

Typical flow:
1) Generate or load an image (numpy array or file path)
2) Detect centroids with BlobDetection
3) Estimate grid rotation (optional) and rectify
4) Assign centroids to grid with fit_grid_and_assign

See atommovr/tests/test_imaging.py for examples.
"""

from .generation import (
    gaussian_2d,
    generate_gaussian_image,
    generate_gaussian_image_from_binary_grid,
)

from .geometry import rotate_points_ccw, rotate_points_cw

from .extraction import (
    Extractor,
    BlobDetection,
    fit_grid_and_assign,
    rotate_image,
    inverse_rotate_centroids,
    estimate_grid_rotation_pca,
    estimate_grid_rotation_vectorize,
    estimate_grid_rotation_diff_pca,
    estimate_grid_rotation_diffs,
    estimate_grid_rotation_pair_diff,
    estimate_grid_rotation_fit_rect,
    estimate_grid_rotation_fourier_img,
    estimate_grid_rotation_fourier,
)

__all__ = [
    # generation
    "gaussian_2d",
    "generate_gaussian_image",
    "generate_gaussian_image_from_binary_grid",
    "rotate_points_ccw",
    "rotate_points_cw",
    # extraction classes
    "Extractor",
    "BlobDetection",
    # extraction functions
    "fit_grid_and_assign",
    "rotate_image",
    "inverse_rotate_centroids",
    "estimate_grid_rotation_pca",
    "estimate_grid_rotation_vectorize",
    "estimate_grid_rotation_diff_pca",
    "estimate_grid_rotation_diffs",
    "estimate_grid_rotation_pair_diff",
    "estimate_grid_rotation_fit_rect",
    "estimate_grid_rotation_fourier_img",
    "estimate_grid_rotation_fourier",
    # viz
    "overlay_centroids",
]

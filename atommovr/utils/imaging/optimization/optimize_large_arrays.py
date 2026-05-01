"""Extended blob-detector hyper-parameter sweep for very large arrays.

This script generates synthetic images with thousands of target sites, runs
SimpleBlobDetector-based extraction, estimates rotation, assigns the centroids
back to the grid and tracks feasibility (recall/precision >= 0.99) for each
parameter combination.

Outputs
-------
- figs/benchmark_pipeline/large_array_param_search.csv : per-parameter summary
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from atommovr.utils.imaging.extraction import (
    BlobDetection,
    estimate_grid_rotation_fit_rect,
    fit_grid_and_assign,
    inverse_rotate_centroids,
)
from atommovr.utils.imaging.generation import generate_gaussian_image
from atommovr.utils.imaging.optimization._shared import (
    _angle_error_deg,
    _compute_assignment_metrics,
    _rotate_points_about_center,
    _sample_sparse_grid_points,
)


@dataclass
class Sample:
    image: np.ndarray
    binary: np.ndarray
    angle_deg: float
    image_shape: Tuple[int, int]


def _generate_samples(
    n_samples: int,
    grid_size: int,
    image_shape: Tuple[int, int],
    angles: Sequence[float],
    load_probability: float,
) -> List[Sample]:
    samples: List[Sample] = []
    for idx in range(n_samples):
        rng = np.random.default_rng(idx)
        points, binary, row_spacing, col_spacing = _sample_sparse_grid_points(
            grid_size=grid_size,
            image_shape=image_shape,
            load_probability=load_probability,
            rng=rng,
        )
        if len(points) == 0:
            continue

        jitter = np.zeros_like(points)
        jitter[:, 0] = rng.normal(scale=0.05 * row_spacing, size=len(points))
        jitter[:, 1] = rng.normal(scale=0.05 * col_spacing, size=len(points))
        jittered_points = points + jitter

        brightness = rng.uniform(0.8, 1.0, size=len(points))
        sigmas = rng.normal(loc=1.5, scale=0.05, size=len(points))
        true_angle = float(rng.choice(angles))
        rotated_points = _rotate_points_about_center(
            jittered_points,
            angle_deg=true_angle,
            image_shape=image_shape,
        )
        img = generate_gaussian_image(
            points=rotated_points.tolist(),
            sigmas=sigmas,
            brightness_factors=brightness,
            image_shape=image_shape,
            angle=0.0,
        )

        # Normalize and add mild noise stripes to mimic camera artifacts
        img = img - img.min()
        img = img / (np.ptp(img) + 1e-8)
        noise = rng.normal(loc=0.0, scale=0.01, size=img.shape)
        img = np.clip(img + noise, 0.0, 1.0)
        img_u8 = (img * 255).astype(np.uint8)
        if cv2 is not None:
            img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = np.stack([img_u8, img_u8, img_u8], axis=-1)
        samples.append(
            Sample(
                image=img_bgr,
                binary=binary,
                angle_deg=true_angle,
                image_shape=img_bgr.shape[:2],
            )
        )
    return samples


def _evaluate_params(
    samples: Iterable[Sample],
    params: Dict[str, float],
    grid_shape: Tuple[int, int],
) -> Dict[str, float]:
    detector = BlobDetection(shape=grid_shape, logger=None, blob_params=None)
    for k, v in params.items():
        setattr(detector.blob_params, k, v)

    times_ms: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []
    feasible_flags: List[bool] = []
    angle_errors: List[float] = []

    for sample in samples:
        start = time.perf_counter()
        centroids, _ = detector.extract(sample.image)
        detect_time = (time.perf_counter() - start) * 1000.0
        times_ms.append(detect_time)

        if len(centroids) == 0:
            feasible_flags.append(False)
            recalls.append(0.0)
            precisions.append(0.0)
            angle_errors.append(np.nan)
            continue

        try:
            est_angle = estimate_grid_rotation_fit_rect(centroids, plot=False)
            centroids_corr = inverse_rotate_centroids(
                centroids=np.asarray(centroids),
                image_shape=sample.image_shape,
                angle_deg=est_angle,
            )
            assigned = fit_grid_and_assign(
                centroids_corr,
                grid_shape,
                image_shape=sample.image_shape,
            )
            stats = _compute_assignment_metrics(assigned, sample.binary)
            feasible = stats["recall"] >= 0.99 and stats["precision"] >= 0.99
            recalls.append(stats["recall"])
            precisions.append(stats["precision"])
            feasible_flags.append(feasible)
            angle_diff = abs(_angle_error_deg(est_angle, sample.angle_deg))
            angle_errors.append(angle_diff)
        except Exception:
            feasible_flags.append(False)
            recalls.append(0.0)
            precisions.append(0.0)
            angle_errors.append(np.nan)

    return {
        "avg_time_ms": float(np.mean(times_ms)),
        "std_time_ms": float(np.std(times_ms)),
        "mean_recall": float(np.mean(recalls)),
        "mean_precision": float(np.mean(precisions)),
        "feasibility_rate": float(np.mean(feasible_flags)),
        "mean_angle_error": float(np.nanmean(angle_errors)),
    }


def run_large_array_optimization():
    grid_size = 45  # 45x45 = 2025 target sites
    image_shape = (1200, 1200)
    load_probability = 0.75
    n_samples = 6
    angles = (-10, -5, -2, 0, 2, 5, 10)

    print(
        f"Generating {n_samples} synthetic images with grid {grid_size}x{grid_size}"
        f" (~{grid_size * grid_size} sites)..."
    )
    samples = _generate_samples(
        n_samples=n_samples,
        grid_size=grid_size,
        image_shape=image_shape,
        angles=angles,
        load_probability=load_probability,
    )

    if not samples:
        raise RuntimeError("Failed to generate synthetic samples for optimization.")

    param_grid = {
        "minThreshold": [60, 80, 100],
        "maxThreshold": [255],
        "thresholdStep": [5, 10, 20],
        "minDistBetweenBlobs": [5, 10, 20],
        "minArea": [5, 10, 20],
        "filterByColor": [True],
        "blobColor": [255],
        "filterByArea": [False, True],
        "filterByCircularity": [False],
        "filterByConvexity": [False],
        "filterByInertia": [False],
    }

    keys, values = zip(*param_grid.items(), strict=True)
    combinations = [
        dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)
    ]
    print(f"Evaluating {len(combinations)} parameter combinations...")

    results: List[Dict[str, float]] = []
    grid_shape_tuple = (grid_size, grid_size)
    for idx, combo in enumerate(combinations):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(combinations)}")
        metrics = _evaluate_params(samples, combo, grid_shape_tuple)
        results.append({**combo, **metrics})

    df = pd.DataFrame(results)
    output_dir = os.path.join("figs", "benchmark_pipeline")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "large_array_param_search.csv")
    df.to_csv(output_csv, index=False)
    print(f"Wrote detailed results to {output_csv}")

    feasible_only = df[df["feasibility_rate"] >= 0.99]
    if feasible_only.empty:
        print("No parameter set achieved >=99% feasibility. Showing best overall:")
        print(
            df.sort_values(["feasibility_rate", "mean_recall"], ascending=False).head(5)
        )
    else:
        print("Top feasible parameter sets (>=99% feasibility):")
        cols = [
            "minThreshold",
            "thresholdStep",
            "minDistBetweenBlobs",
            "minArea",
            "filterByArea",
            "avg_time_ms",
            "feasibility_rate",
            "mean_recall",
            "mean_precision",
        ]
        print(feasible_only.sort_values(["avg_time_ms"]).loc[:, cols].head(5))


if __name__ == "__main__":
    run_large_array_optimization()

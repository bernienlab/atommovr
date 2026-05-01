"""Blob-detector parameter sweep aligned with ``test_grid_extraction``.

This script mirrors the data-generation and evaluation strategy from
``test_grid_extraction`` so we can study SimpleBlobDetector hyper-parameters on
the large 1200x1200 scenes with ~45x45 grids that routinely appear in the
pipeline.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from atommovr.utils.imaging.extraction import BlobDetection
from atommovr.utils.imaging.optimization._shared import (
    _compute_assignment_metrics,
    generate_rot_img,
    setup_blob_params,
)


@dataclass
class GridSample:
    image_path: str
    grid_shape: Tuple[int, int]
    true_binary: np.ndarray
    n_points: int


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("optimize_blob_params")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _prepare_samples(
    seeds: Sequence[int],
    grid_sizes: Sequence[int],
    image_shape: Tuple[int, int],
    output_dir: Path,
) -> List[GridSample]:
    """Generate large-grid samples via ``generate_rot_img`` just like the tests."""

    samples: List[GridSample] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed, grid_size in zip(seeds, grid_sizes, strict=True):
        np.random.seed(seed)
        suffix = f"blobopt_seed{seed}_grid{grid_size}"
        points, true_binary = generate_rot_img(
            image_shape=image_shape,
            grid_size=grid_size,
            true_angle=0.0,
            suffix=suffix,
            directory=str(output_dir),
        )
        image_path = output_dir / f"{suffix}_image.png"
        samples.append(
            GridSample(
                image_path=str(image_path),
                grid_shape=(grid_size, grid_size),
                true_binary=true_binary,
                n_points=len(points),
            )
        )
    return samples


def _evaluate_params(
    samples: Iterable[GridSample],
    params: Dict[str, float],
    logger: logging.Logger,
) -> Dict[str, float]:
    recalls: List[float] = []
    precisions: List[float] = []
    feasibility: List[bool] = []
    exact_matches = 0
    sample_list = list(samples)

    start = time.perf_counter()
    for sample in sample_list:
        blob_params = setup_blob_params(params)
        detector = BlobDetection(
            shape=sample.grid_shape,
            spots=sample.n_points,
            scale=(1, 1),
            logger=logger,
            blob_params=blob_params,
        )
        try:
            binary = detector.extract_estimate_rotate_and_assign(
                sample.image_path, visualize=False
            )
        except Exception as exc:  # pragma: no cover - guardrail for bad params
            logger.debug("Extraction failed for %s: %s", sample.image_path, exc)
            recalls.append(0.0)
            precisions.append(0.0)
            feasibility.append(False)
            continue

        stats = _compute_assignment_metrics(binary, sample.true_binary)
        recalls.append(stats["recall"])
        precisions.append(stats["precision"])
        exact_matches += int(stats["exact_match"])
        feasibility.append(stats["recall"] >= 0.99 and stats["precision"] >= 0.99)

    avg_time_ms = (time.perf_counter() - start) / max(len(sample_list), 1) * 1000.0
    return {
        "avg_time_ms": avg_time_ms,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "feasibility_rate": float(np.mean(feasibility)) if feasibility else 0.0,
        "exact_match_rate": exact_matches / len(sample_list) if sample_list else 0.0,
    }


def run_benchmark():
    logger = _get_logger()
    image_shape = (1200, 1200)
    seeds = range(45, 50)
    grid_sizes = range(45, 50)
    samples = _prepare_samples(
        seeds, grid_sizes, image_shape, Path("figs/blob_param_opt")
    )

    param_grid = {
        "minThreshold": [70, 80, 100],
        "maxThreshold": [255],
        "thresholdStep": [15, 20],
        "minDistBetweenBlobs": [5, 10, 15, 20],
        "minArea": [5, 10, 15],
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
    logger.info("Evaluating %d parameter combinations...", len(combinations))

    results: List[Dict[str, float]] = []
    for idx, combo in enumerate(combinations):
        if idx % 10 == 0:
            logger.info("Progress: %d/%d", idx, len(combinations))
        stats = _evaluate_params(samples, combo, logger)
        results.append({**combo, **stats})

    df = pd.DataFrame(results)
    output_dir = Path("figs/benchmark_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "blob_param_search_grid_extraction.csv"
    df.to_csv(output_csv, index=False)
    logger.info("Wrote results to %s", output_csv)

    feasible = df[df["feasibility_rate"] >= 0.99]
    if feasible.empty:
        logger.warning("No configuration reached 99%% feasibility. Showing top rows:")
        print(
            df.sort_values(
                ["feasibility_rate", "mean_recall", "avg_time_ms"],
                ascending=[False, False, True],
            )[
                [
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
            ].head(
                5
            )
        )
    else:
        logger.info("Top feasible configurations (>=99%% feasibility):")
        print(
            feasible.sort_values("avg_time_ms")[
                [
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
            ].head(5)
        )


if __name__ == "__main__":
    run_benchmark()

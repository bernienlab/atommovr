import numpy as np
import pytest
from atommovr.utils.imaging.extraction import (
    BlobDetection,
    fit_grid_and_assign,
    inverse_rotate_centroids,
    rotate_image,
    estimate_grid_rotation_diffs,
    estimate_grid_rotation_pca,
    estimate_grid_rotation_pair_diff,
    estimate_grid_rotation_diff_pca,
    estimate_grid_rotation_vectorize,
    estimate_grid_rotation_fit_rect,
    estimate_grid_rotation_fourier_img,
    estimate_grid_rotation_fourier,
)

from atommovr.utils.imaging.generation import (
    generate_gaussian_image_from_binary_grid,
    generate_gaussian_image,
    compute_scaled_image_shape,
)
from atommovr.utils.imaging.geometry import rotate_points_ccw


import matplotlib.pyplot as plt
import logging
import cv2
import itertools
import os
import time
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment

from typing import Optional, Tuple, List, Callable, Any, Sequence

logger = logging.getLogger(__name__)


@pytest.fixture(name="logger")
def logger():
    """Provide a simple logger fixture for tests that request `logger`.

    The original tests annotated the `logger` parameter with a `logging.Logger`
    type but relied on a pytest fixture. Add `logger_fixture` and alias it to
    the expected name via pytest's fixture mechanism below.
    """
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("atommovr.tests.test_imaging")


def test_round_trip_small_grid(tmp_path):
    grid = np.zeros((5, 7), dtype=int)
    grid[1, 1] = 1
    grid[2, 3] = 1
    grid[4, 6] = 1
    img = generate_gaussian_image_from_binary_grid(
        grid,
        sigma=1.2,
        brightness_factor=1.0,
        image_shape=(128, 128),
        noise_level=0.01,
        stripe_intensity=0.0,
    )
    # Write to disk for OpenCV blob detector path
    img_u8 = (255 * (img - img.min()) / (np.ptp(img) + 1e-8)).astype(np.uint8)
    p = tmp_path / "synthetic.png"
    import imageio

    imageio.imwrite(p, img_u8)

    det = BlobDetection(shape=grid.shape)
    cents, shape = det.extract(str(p))
    assert len(cents) >= grid.sum() - 1  # allow 1 miss in noisy case

    angle = estimate_grid_rotation_pca(cents)
    assert -90 <= angle < 90

    binary = fit_grid_and_assign(cents, grid.shape, image_shape=shape)
    assert binary.shape == grid.shape


def setup_blob_params(
    params: Optional[dict],
    image_shape: Optional[Tuple[int, int]] = None,
    grid_size: Optional[int] = None,
) -> cv2.SimpleBlobDetector_Params:
    if params is None:

        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.filterByColor = True
        blob_params.blobColor = 255
        blob_params.minThreshold = 70
        blob_params.maxThreshold = 255
        blob_params.thresholdStep = 20
        blob_params.minDistBetweenBlobs = 10
        blob_params.minArea = 5
        blob_params.maxArea = 1000
        blob_params.filterByArea = True
        blob_params.filterByCircularity = False
        blob_params.filterByConvexity = False
        blob_params.filterByInertia = False
    else:
        blob_params = cv2.SimpleBlobDetector_Params()
        for key, val in params.items():
            setattr(blob_params, key, val)

    if image_shape is not None and grid_size:
        min_dim = float(min(int(image_shape[0]), int(image_shape[1])))
        spacing = min_dim / max(grid_size + 1, 1)
        blob_params.minDistBetweenBlobs = float(max(2.0, 0.5 * spacing))
        # Loosen thresholds for small synthetic images so blobs are detected reliably
        if min_dim <= 512:
            blob_params.minThreshold = 20
            blob_params.thresholdStep = 10
            blob_params.minArea = 2
            blob_params.maxArea = max(50, int((0.6 * spacing) ** 2))

    return blob_params


def test_grid_extraction(logger: logging.Logger) -> None:
    logger.info("Starting grid extraction tests...")

    test_seeds = range(5, 10)
    grid_sizes = range(5, 10)
    image_shape = (1200, 1200)

    for seed, grid_size in zip(test_seeds, grid_sizes):
        np.random.seed(seed)
        points, true_binary = generate_rot_img(
            image_shape,
            grid_size,
            true_angle=0,
            suffix="test",
            directory="figs/imaging_test",
        )

        # Set up blob detector params
        blob_params = setup_blob_params(None)

        extractor_blob = BlobDetection(
            shape=(grid_size, grid_size),
            spots=len(points),
            scale=(1, 1),
            logger=logger,
            blob_params=blob_params,
        )

        # Extract and compare for BlobDetection
        binary_blob = extractor_blob.extract_estimate_rotate_and_assign(
            "figs/imaging_test/test_image.png", visualize=False
        )

        correct = np.array_equal(binary_blob, true_binary)
        logger.info(f"BlobDetection correct: {correct}")
        if not correct:
            incorrect_coords = np.argwhere(binary_blob != true_binary)
            logger.debug(f"Incorrect blob coordinates: {incorrect_coords}")
            for coord in incorrect_coords:
                logger.debug(
                    f"At {coord}, expected {true_binary[coord[0]][coord[1]]} but got {binary_blob[coord[0]][coord[1]]}"
                )
    logger.info("Finished grid extraction tests.")


def generate_ideal_grid(
    grid_shape: Tuple[int, int], image_shape: Tuple[int, int] = (640, 1280)
) -> np.ndarray:
    """
    Generate ideal grid coordinates that match the placement logic in
    `generate_rot_img` (so start and spacing are consistent).

    Returns coordinates in (y, x) order.
    """
    # start positions use integer division to match generate_rot_img
    start_x = image_shape[0] // grid_shape[0]
    start_y = image_shape[1] // grid_shape[1]
    spacing_x = image_shape[0] // (grid_shape[0] + 1)
    spacing_y = image_shape[1] // (grid_shape[1] + 1)
    ideal_grid = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            ideal_grid.append((start_x + i * spacing_x, start_y + j * spacing_y))
    return np.array(ideal_grid)


def test_estimation_and_extraction(logger: logging.Logger) -> None:
    logger.info("Starting extraction tests...")

    test_seeds = range(20, 21)
    grid_sizes = range(20, 21)
    image_shape = (640, 1280)

    # Define hyperparameter grid
    param_grid = {
        "filterByColor": [True],
        "blobColor": [255],
        "minThreshold": [120],
        "maxThreshold": [255],
        "thresholdStep": [5],
        "minDistBetweenBlobs": [30],
        "minArea": [20],
        "maxArea": [1000],
        "filterByArea": [False],
        "filterByCircularity": [False],
        "minCircularity": [0.9],
        "filterByConvexity": [False],
        "minConvexity": [0.8],
        "filterByInertia": [False],
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = 0
    best_params: Optional[dict] = None

    # --- Calibration step: generate a calibration image and estimate rotation ---
    calib_seed = 42
    calib_grid_size = 7
    np.random.seed(calib_seed)
    true_angles = [10]  # [-15, -10, -5, 0, 5, 10, 15]

    for true_angle in true_angles:
        points, _ = generate_rot_img(
            image_shape,
            calib_grid_size,
            true_angle,
            suffix="calib",
            directory="figs/imaging_test",
        )

        # clashes with parameter grid
        blob_params = setup_blob_params(None)

        # Use BlobDetection to extract centroids from calibration image
        extractor_blob = BlobDetection(
            shape=(calib_grid_size, calib_grid_size),
            scale=(1, 1),
            logger=logger,
            blob_params=blob_params,
        )
        calib_centroids, _ = extractor_blob.extract(
            "figs/imaging_test/calib_rot_image.png"
        )

        # circle centroids in red on calibration image
        # Read the image with matplotlib, convert to uint8 BGR for OpenCV drawing
        img_calib = plt.imread("figs/imaging_test/calib_rot_image.png")
        # If the image is normalized float [0,1], convert to uint8
        if img_calib.dtype == np.float32 or img_calib.dtype == np.float64:
            img_disp = np.clip(img_calib * 255.0, 0, 255).astype(np.uint8)
        else:
            img_disp = img_calib.copy()

        # If image is single-channel grayscale, convert to BGR so we can draw colored circles
        if img_disp.ndim == 2:
            img_bgr = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
        elif img_disp.shape[2] == 4:
            # RGBA -> RGB
            img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_RGBA2RGB)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            # RGB -> BGR
            img_bgr = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)

        # Draw unfilled red circles (thickness 2) at centroid locations (x=col, y=row)
        for y_c, x_c in calib_centroids:
            center = (int(round(x_c)), int(round(y_c)))
            cv2.circle(img_bgr, center, radius=10, color=(0, 0, 255), thickness=2)

        # Save using OpenCV which preserves resolution; convert back to RGB for matplotlib-friendly files if desired
        os.makedirs(
            os.path.dirname("figs/imaging_test/centroids_on_calib.png"), exist_ok=True
        )
        cv2.imwrite("figs/imaging_test/centroids_on_calib.png", img_bgr)

        # Estimate rotation angle
        rotation_angle_diffs = estimate_grid_rotation_diffs(calib_centroids, plot=True)

        rotation_angle_pca = estimate_grid_rotation_pca(calib_centroids, plot=True)

        rotation_angle_pair_diff = estimate_grid_rotation_pair_diff(
            centroids=calib_centroids, plot=True
        )

        rotation_angle_diff_pca = estimate_grid_rotation_diff_pca(
            calib_centroids, plot=True
        )

        rotation_angle_vectorize = estimate_grid_rotation_vectorize(
            calib_centroids, (calib_grid_size, calib_grid_size), plot=True
        )

        rotation_angle_fourier = estimate_grid_rotation_fourier(
            calib_centroids, image_shape=image_shape, plot=True
        )

        rotation_angle_fourier_img = estimate_grid_rotation_fourier_img(
            plt.imread("figs/imaging_test/calib_rot_image.png"), plot=True
        )

        rotation_angle_rect_fit = estimate_grid_rotation_fit_rect(
            calib_centroids, plot=True
        )

        rotation_angle = rotation_angle_rect_fit  # np.deg2rad(true_angle)
        logger.info(
            f"Estimated grid rotation angle (degrees): \n"
            f"Diffs: {rotation_angle_diffs}, \n "
            f"PCA: {rotation_angle_pca}, \n "
            f"Pair-diffs: {rotation_angle_pair_diff}, \n"
            f"rotation_angle_diff_pca {rotation_angle_diff_pca} \n"
            f"rotation_angle_vectorize {rotation_angle_vectorize} \n"
            f"rotation_angle_fourier {rotation_angle_fourier} \n"
            f"rotation_angle_fourier_img {rotation_angle_fourier_img} \n"
            f"Rect fit: {rotation_angle_rect_fit} (preferred), \n"
            f"true angle {true_angle} degrees"
        )

        img = plt.imread("figs/imaging_test/calib_rot_image.png")

        for param_set in param_combinations:
            logger.info(f"Testing blob params: {param_set}")
            local_blob_correct = 0
            local_runs = 0

            for seed, grid_size in zip(test_seeds, grid_sizes):
                np.random.seed(seed)
                points, true_binary = generate_rot_img(
                    image_shape,
                    grid_size,
                    true_angle,
                    suffix="test",
                    directory="figs/imaging_test",
                )

                # Read it back using matplotlib
                img = plt.imread("figs/imaging_test/test_rot_image.png")

                img_back_rot = rotate_image(img, -rotation_angle)

                # Save the corrected image
                cor_rot_image_path = "figs/imaging_test/cor_rot_image.png"

                plt.imsave(cor_rot_image_path, img_back_rot, cmap="Blues")

                # Set up blob detector params
                blob_params = setup_blob_params(params=param_set)

                extractor_blob = BlobDetection(
                    shape=(grid_size, grid_size),
                    spots=len(points),
                    scale=(1, 1),
                    logger=logger,
                    blob_params=blob_params,
                )

                # Extract and compare for BlobDetection on the rotated image
                binary_blob = extractor_blob.extract_estimate_rotate_and_assign(
                    cor_rot_image_path, visualize=True
                )

                correct = np.array_equal(binary_blob, true_binary)
                logger.info(f"BlobDetection correct: {correct}")
                if not correct:
                    incorrect_coords = np.argwhere(binary_blob != true_binary)
                    logger.debug(f"Incorrect blob coordinates: {incorrect_coords}")
                    for coord in incorrect_coords:
                        logger.debug(
                            f"At {coord}, expected {true_binary[coord[0]][coord[1]]} but got {binary_blob[coord[0]][coord[1]]}"
                        )
                local_blob_correct += int(correct)
                local_runs += 1

            logger.info(
                f"BlobDetection correct for this param set: {local_blob_correct}/{local_runs}"
            )

            if local_blob_correct > best_score:
                best_score = local_blob_correct
                best_params = param_set

        logger.info(
            f"Best blob params: {best_params} with score {best_score}/{local_runs}"
        )


def test_vectorize_fit(
    logger: logging.Logger, angles: Optional[List[float]] = None, plot: bool = False
) -> None:
    estimation_method_test(
        logger, estimate_grid_rotation_vectorize, angles=angles, plot=plot
    )


def test_pair_diff_fit(
    logger: logging.Logger, angles: Optional[List[float]] = None, plot: bool = False
) -> None:
    estimation_method_test(
        logger, estimate_grid_rotation_pair_diff, angles=angles, plot=plot
    )


def test_diff_fit(
    logger: logging.Logger, angles: Optional[List[float]] = None, plot: bool = False
) -> None:
    estimation_method_test(
        logger, estimate_grid_rotation_diffs, angles=angles, plot=plot
    )


def test_rect_fit(
    logger: logging.Logger, angles: Optional[List[float]] = None, plot: bool = False
) -> None:
    estimation_method_test(
        logger, estimate_grid_rotation_fit_rect, angles=angles, plot=plot
    )


def test_fourier_estimation(
    logger: logging.Logger, angles: Optional[List[float]] = None, plot: bool = False
) -> None:
    estimation_method_test(
        logger, estimate_grid_rotation_fourier, angles=angles, plot=plot
    )


def test_PCA_estimation(
    logger: logging.Logger, angles: Optional[List[float]] = None, plot: bool = False
) -> None:
    estimation_method_test(logger, estimate_grid_rotation_pca, angles=angles, plot=plot)


def test_PCA_diff_estimation(
    logger: logging.Logger, angles: Optional[List[float]] = None, plot: bool = False
) -> None:
    estimation_method_test(
        logger, estimate_grid_rotation_diff_pca, angles=angles, plot=plot
    )


def estimation_method_test(
    logger: logging.Logger,
    method: Callable[..., float],
    angles: Optional[List[float]] = None,
    path: str = "figs/benchmark_angle_est/",
    plot: bool = False,
) -> None:
    logger.info(f"Starting estimation tests for {method.__name__}...")

    test_seeds = range(10, 11)
    grid_sizes = range(10, 11)
    image_shape = (640, 1280)

    if angles is None:
        true_angles = [-45, -30, -5, -2, 0, 2, 5, 30, 45]
    else:
        true_angles = angles

    records: List[dict] = []

    for seed, grid_size in zip(test_seeds, grid_sizes):
        for true_angle in true_angles:
            np.random.seed(seed)
            points, _ = generate_rot_img(
                image_shape,
                grid_size,
                true_angle=true_angle,
                suffix="test",
                directory=path,
            )

            img = plt.imread(os.path.join(path, "test_image.png"))
            if img.ndim == 3:
                img = img[..., 0]

            # Set up blob detector params, takes default if None
            blob_params = setup_blob_params(params=None)

            extractor_blob = BlobDetection(
                shape=(grid_size, grid_size),
                spots=len(points),
                scale=(1, 1),
                logger=logger,
                blob_params=blob_params,
            )

            centroids, _ = extractor_blob.extract(
                os.path.join(path, "test_rot_image.png")
            )

            # --- Time the estimation ---
            start_time = time.perf_counter()
            if method == estimate_grid_rotation_vectorize:
                angle_est_deg = method(
                    centroids=centroids, grid_shape=(grid_size, grid_size), plot=plot
                )
            elif method == estimate_grid_rotation_fourier_img:
                angle_est_deg = method(img=img, plot=plot)
            elif method == estimate_grid_rotation_fourier:
                angle_est_deg = method(
                    centroids=centroids, image_shape=image_shape, plot=plot
                )
            else:
                angle_est_deg = method(centroids=centroids, plot=plot)
            end_time = time.perf_counter()

            error = abs(angle_est_deg - true_angle)

            name = (
                method.__name__.replace("estimate_grid_rotation_", "")
                .replace("_", "-")
                .upper()
            )

            records.append(
                {
                    "Method": name,
                    "True Angle": true_angle,
                    "Estimated Angle": angle_est_deg,
                    "Error": error,
                    "Within Tolerance": error < 2,
                    "Time (s)": end_time - start_time,
                }
            )

            logger.info(
                f"True: {true_angle}°, "
                f"Estimated: {angle_est_deg:.2f}°, "
                f"Error: {error:.2f}°, "
                f"Time: {end_time - start_time:.4f}s"
            )

    # --- Convert to dataframe and store ---
    df = pd.DataFrame(records)
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, "benchmark_estimation.csv")

    # Append if exists, else write with header
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_path, index=False)

    logger.info(f"Benchmark results appended to {csv_path}")


def test_estimation_feasibility(
    logger: logging.Logger,
    angles: Optional[List[float]] = None,
    grid_size: int = 9,
    image_shape: Tuple[int, int] = (400, 400),
    output_csv: str = "data/benchmark_pipeline/feasibility_results_0812.csv",
) -> None:
    """
    For each estimation method and angle, check whether the estimated
    rotation followed by inverse-rotating the extracted centroids leads to a
    feasible assignment to the expected binary grid.

    Saves a CSV with per-method/angle recall and a boolean `Feasible` flag.
    """
    logger.info("Starting feasibility tests for estimation methods...")

    if angles is None:
        angles = [-10, -7, -5, -2, 0, 2, 5, 7, 10]

    methods = [
        # estimate_grid_rotation_diffs,
        # estimate_grid_rotation_pca,
        # estimate_grid_rotation_pair_diff,
        # estimate_grid_rotation_diff_pca,
        # estimate_grid_rotation_vectorize,
        estimate_grid_rotation_fit_rect,
        # estimate_grid_rotation_fourier_img,
        # estimate_grid_rotation_fourier,
    ]

    records: List[dict] = []

    for angle in angles:
        np.random.seed(42)
        points, _ = generate_rot_img(
            image_shape,
            grid_size,
            true_angle=angle,
            suffix=f"feas_{int(angle)}",
            directory="figs/imaging_test/feasability",
        )

        os.makedirs("figs/imaging_test/feasability", exist_ok=True)

        # initial extraction on rotated image
        rot_img_path = f"figs/imaging_test/feasability/feas_{int(angle)}_rot_image.png"
        img = plt.imread(rot_img_path)
        if img.ndim == 3:
            img_gray = img[..., 0]
        else:
            img_gray = img

        blob_params = setup_blob_params(None)
        extractor_blob = BlobDetection(
            shape=(grid_size, grid_size),
            spots=len(points),
            scale=(1, 1),
            logger=logger,
            blob_params=blob_params,
        )
        centroids, _ = extractor_blob.extract(rot_img_path)

        for method in methods:
            name = (
                method.__name__.replace("estimate_grid_rotation_", "")
                .replace("_", "-")
                .upper()
            )
            try:
                if method == estimate_grid_rotation_vectorize:
                    est_angle = method(
                        centroids=centroids,
                        grid_shape=(grid_size, grid_size),
                        plot=False,
                    )
                elif method == estimate_grid_rotation_fourier_img:
                    est_angle = method(img=img_gray, plot=False)
                elif method == estimate_grid_rotation_fourier:
                    est_angle = method(
                        centroids=centroids, image_shape=image_shape, plot=False
                    )
                else:
                    est_angle = method(centroids=centroids, plot=False)
                est_success = True
            except Exception as e:
                logger.debug(f"Estimation {name} failed during feasibility: {e}")
                est_angle = float("nan")
                est_success = False

            feasible = False
            recall = float("nan")

            if est_success and not np.isnan(est_angle) and len(centroids) > 0:
                try:
                    centroids_corrected = inverse_rotate_centroids(
                        centroids=np.asarray(centroids),
                        image_shape=img.shape,
                        angle_deg=est_angle,
                    )
                    # Attempt to assign to grid
                    assigned = fit_grid_and_assign(
                        centroids_corrected,
                        (grid_size, grid_size),
                        image_shape=img.shape[:2],
                    )
                    true_binary = np.zeros((grid_size, grid_size), dtype=int)
                    for pt in points:
                        # Map point to grid index
                        row = int(
                            round(
                                (pt[0] - (image_shape[0] // grid_size))
                                / (image_shape[0] // (grid_size + 1))
                            )
                        )
                        col = int(
                            round(
                                (pt[1] - (image_shape[1] // grid_size))
                                / (image_shape[1] // (grid_size + 1))
                            )
                        )
                        if 0 <= row < grid_size and 0 <= col < grid_size:
                            true_binary[row, col] = 1
                    tp = np.sum((assigned == 1) & (true_binary == 1))
                    fn = np.sum((assigned == 0) & (true_binary == 1))
                    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                    feasible = recall >= 0.99  # at least 99% recall to be feasible

                    feasible = true_binary == assigned
                    feasible = feasible.all()

                except Exception as e:
                    logger.debug(f"Feasibility check failed for {name}: {e}")

            records.append(
                {
                    "Method": name,
                    "Angle": angle,
                    "Estimation Success": est_success,
                    "Estimated Angle (deg)": est_angle,
                    "Feasible": feasible,
                    "Recall": recall,
                }
            )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote feasibility results to {output_csv}")


def make_plots(
    df: Optional[pd.DataFrame] = None,
    source_csv: Optional[str] = None,
    save_dir: str = "figs/benchmark_plots",
) -> None:
    """
    Generate plots from the benchmark dataframe or CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing benchmark results.
    source_csv : str
        Path to CSV file containing benchmark results.
    save_dir : str
        Directory to save the plots.

    Returns
    -------
        None
    """

    if df is None and source_csv is not None:
        df = pd.read_csv(source_csv)
    elif df is None:
        raise ValueError("Either df or source_csv must be provided.")

    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # --- 1. True vs. Estimated Angles ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="True Angle",
        y="Estimated Angle",
        hue="Method",
        style="Method",
        s=100,
    )
    plt.plot([-50, 50], [-50, 50], "k--", lw=2)  # ideal line
    plt.xlabel("True Angle (deg)")
    plt.ylabel("Estimated Angle (deg)")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/true_vs_estimated.svg", format="svg", dpi=200)
    plt.close()

    # --- 2. Error vs. True Angle ---
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="True Angle", y="Error", hue="Method", errorbar="sd")
    plt.xlabel("True Angle (deg)")
    plt.ylabel("Absolute Error (deg)")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_vs_angle.svg", format="svg", dpi=200)
    plt.close()

    # --- 3. Error distribution ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Method", y="Error")
    sns.swarmplot(data=df, x="Method", y="Error", color=".25")
    # plt.title("Error Distribution per Method")
    plt.xlabel("Method")
    plt.ylabel("Absolute Error (deg)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_distribution.svg", format="svg", dpi=200)
    plt.close()

    print(f"Plots saved to {save_dir}")


def generate_rot_img_in_memory(
    image_shape: Tuple[int, int],
    grid_size: int,
    true_angle: float,
    *,
    save_images: bool = False,
    suffix: Optional[str] = None,
    directory: str = "output",
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, Optional[str]]:
    """
    Generate a rotated Gaussian grid image and optionally save it.

    Returns
    -------
    points : List[Tuple[int, int]]
        List of (x, y) coordinates of the Gaussian centers.
    true_binary : np.ndarray
        Binary grid indicating presence (1) or absence (0) of Gaussians.
    rot_image : np.ndarray
        Rotated image array.
    rot_image_path : Optional[str]
        Path to the saved rotated image if requested.
    """
    image_shape = compute_scaled_image_shape(image_shape, grid_size)
    grid_shape = (grid_size, grid_size)
    points: List[Tuple[int, int]] = []
    start_x = image_shape[0] // grid_shape[0]
    start_y = image_shape[1] // grid_shape[1]
    spacing_x = image_shape[0] // (grid_shape[0] + 1)
    spacing_y = image_shape[1] // (grid_shape[1] + 1)
    true_binary = np.zeros(grid_shape, dtype=int)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if np.random.rand() > 0.4:
                points.append((start_x + i * spacing_x, start_y + j * spacing_y))
                true_binary[i, j] = 1

    sigmas = np.random.normal(1.5, 0.01, len(points))
    brightness_factors = np.random.uniform(0.8, 1.0, len(points))

    gaussian_img = generate_gaussian_image(
        points, sigmas, brightness_factors, image_shape, angle=0.0
    )
    rot_image = generate_gaussian_image(
        points, sigmas, brightness_factors, image_shape, angle=true_angle
    )

    rot_path = None
    if save_images:
        if suffix is None:
            raise ValueError("suffix must be provided when save_images=True")
        os.makedirs(directory, exist_ok=True)

        import imageio.v2 as imageio

        def _to_uint8(img: np.ndarray) -> np.ndarray:
            arr = np.asarray(img)
            if arr.dtype == np.uint8:
                return arr
            arr = arr.astype(np.float32, copy=False)
            min_val = float(np.min(arr)) if arr.size else 0.0
            max_val = float(np.max(arr)) if arr.size else 0.0
            if max_val <= min_val:
                return np.zeros(arr.shape, dtype=np.uint8)
            arr = (arr - min_val) / (max_val - min_val)
            return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        imageio.imwrite(f"{directory}/{suffix}_image.png", _to_uint8(gaussian_img))
        rot_path = f"{directory}/{suffix}_rot_image.png"
        imageio.imwrite(rot_path, _to_uint8(rot_image))
        print(f"Saved images to {directory}/{suffix}_image.png and {rot_path}")
    return points, true_binary, rot_image, rot_path


def generate_rot_img(
    image_shape: Tuple[int, int],
    grid_size: int,
    true_angle: float,
    suffix: Optional[str],
    directory: str = "output",
) -> Tuple[Tuple[int, int], List[Tuple[int, int]], np.ndarray]:
    """
    Generate a rotated Gaussian grid image and save it.
    """
    if suffix is None:
        raise ValueError("suffix must be provided for generate_rot_img")
    points, true_binary, _rot_image, _ = generate_rot_img_in_memory(
        image_shape,
        grid_size,
        true_angle,
        save_images=True,
        suffix=suffix,
        directory=directory,
    )
    return points, true_binary


def _sample_sparse_grid_points(
    grid_size: int,
    image_shape: Tuple[int, int],
    load_probability: float = 0.6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate sparse grid coordinates matching generate_rot_img spacing."""
    if rng is None:
        rng = np.random.default_rng()
    image_shape = compute_scaled_image_shape(image_shape, grid_size)
    grid_shape = (grid_size, grid_size)
    img_h, img_w = image_shape[:2]
    start_row = float(img_h // grid_shape[0])
    start_col = float(img_w // grid_shape[1])
    row_spacing = float(img_h // (grid_shape[0] + 1))
    col_spacing = float(img_w // (grid_shape[1] + 1))
    points: List[Tuple[float, float]] = []
    binary = np.zeros(grid_shape, dtype=int)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if rng.random() < load_probability:
                points.append(
                    (start_row + i * row_spacing, start_col + j * col_spacing)
                )
                binary[i, j] = 1
    return np.asarray(points, dtype=float), binary, row_spacing, col_spacing


def _rotate_points_about_center(
    points: np.ndarray, angle_deg: float, image_shape: Tuple[int, int]
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(-1, 2)
    return rotate_points_ccw(points, image_shape, angle_deg)


def _wrap_angle_deg(angle: float) -> float:
    return (angle + 90.0) % 180.0 - 90.0


def _angle_error_deg(estimate: float, truth: float) -> float:
    return _wrap_angle_deg(estimate - truth)


def _compute_assignment_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=int)
    truth = np.asarray(truth, dtype=int)
    truth_ones = int(truth.sum())
    pred_ones = int(pred.sum())
    true_pos = int(np.logical_and(pred == 1, truth == 1).sum())
    recall = true_pos / truth_ones if truth_ones else 1.0
    precision = true_pos / pred_ones if pred_ones else 1.0
    return {
        "recall": recall,
        "precision": precision,
        "exact_match": bool(np.array_equal(pred, truth)),
        "true_positives": true_pos,
        "true_ones": truth_ones,
        "predicted_ones": pred_ones,
    }


def visualize_result_techniques(
    methods: List[Callable[..., Any]],
    true_angle: float,
    image_shape: Tuple[int, int] = (640, 1280),
) -> None:
    """
    Visualize images with atoms with ground truth angle vs each estimation technique.

    Parameters
    ----------
    methods : List[Callable[..., Any]]
        List of estimation methods to visualize.
    true_angle : float
        Ground truth rotation angle in degrees.
    image_shape : Tuple[int, int]
        Size of the generated image (height, width).

    Returns
    -------
        None
    """
    # Generate test image
    grid_size = 9
    np.random.seed(42)
    points, _ = generate_rot_img(
        image_shape,
        grid_size,
        true_angle=true_angle,
        suffix="test",
        directory="figs/imaging_test",
    )
    rot_img = plt.imread("figs/imaging_test/test_rot_image.png")

    for method in methods:
        centroids = None
        blob_params = setup_blob_params(None)
        extractor_blob = BlobDetection(
            shape=(grid_size, grid_size),
            spots=len(points),
            scale=(1, 1),
            logger=logger,
            blob_params=blob_params,
        )
        centroids, _ = extractor_blob.extract("figs/imaging_test/test_rot_image.png")
        if method == estimate_grid_rotation_vectorize:
            angle_est_deg = method(
                centroids=centroids, grid_shape=(grid_size, grid_size), plot=False
            )
        elif method == estimate_grid_rotation_fourier_img:
            angle_est_deg = method(img=rot_img, plot=False)
        elif method == estimate_grid_rotation_fourier:
            angle_est_deg = method(
                centroids=centroids, image_shape=image_shape, plot=False
            )
        else:
            angle_est_deg = method(centroids=centroids, plot=False)
        logger.info(
            f"Method {method.__name__} estimated angle: {angle_est_deg:.2f} degrees"
        )

        os.makedirs("figs/imaging_test/est_rot", exist_ok=True)

        # Rotate back
        rot_points = inverse_rotate_centroids(
            centroids=centroids,
            image_shape=rot_img.shape,
            angle_deg=angle_est_deg,
        )

        back_rot_img = generate_gaussian_image(
            points=rot_points,
            sigmas=[1.0] * len(rot_points),
            brightness_factors=[1.0] * len(rot_points),
            image_shape=rot_img.shape[:2],
            angle=-angle_est_deg,
        )

        plt.imsave(
            f"figs/imaging_test/est_rot/{method.__name__}_back_rot_image.png",
            back_rot_img,
            cmap="Blues",
        )


def benchmark_rotation_error_feasibility(
    logger: logging.Logger,
    grid_sizes: Sequence[int] = (9, 15, 21, 27, 31),
    angles: Sequence[float] = (-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10),
    seeds: Sequence[int] = tuple(range(10)),
    load_probability: float = 0.60,
    jitter_std: float = 0.05,
    image_shape: Tuple[int, int] = (640, 1280),
    output_csv: str = "data/benchmark_pipeline/rotation_error_impact.csv",
) -> None:
    """Benchmark how rotation estimation errors impact assignment feasibility."""

    logger.info("Starting rotation error propagation benchmark (PCA vs. fit-rect)...")
    methods = {
        "PCA": lambda pts: estimate_grid_rotation_pca(pts, plot=False),
        "FIT_RECT": lambda pts: estimate_grid_rotation_fit_rect(pts, plot=False),
    }

    records: List[dict] = []

    for grid_size in grid_sizes:
        for true_angle in angles:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                base_points, true_binary, row_spacing, col_spacing = (
                    _sample_sparse_grid_points(
                        grid_size,
                        image_shape,
                        load_probability=load_probability,
                        rng=rng,
                    )
                )
                loaded_sites = int(true_binary.sum())
                if loaded_sites < max(4, grid_size // 2):
                    continue

                rotated = _rotate_points_about_center(
                    base_points, true_angle, image_shape
                )
                if jitter_std > 0:
                    rotated = rotated + rng.normal(0.0, jitter_std, size=rotated.shape)

                gt_aligned = inverse_rotate_centroids(
                    rotated, image_shape=image_shape, angle_deg=true_angle
                )
                if len(gt_aligned) == 0:
                    continue

                baseline_binary = fit_grid_and_assign(
                    gt_aligned, (grid_size, grid_size), image_shape=image_shape
                )
                baseline_stats = _compute_assignment_metrics(
                    baseline_binary, true_binary
                )

                center = np.array(
                    [image_shape[0] / 2.0, image_shape[1] / 2.0], dtype=float
                )
                radii = np.linalg.norm(gt_aligned - center, axis=1)
                avg_spacing = (
                    float(row_spacing + col_spacing) / 2.0
                    if (row_spacing + col_spacing)
                    else np.nan
                )

                for method_name, estimator in methods.items():
                    try:
                        est_angle = float(estimator(rotated))
                        est_success = True
                    except Exception as exc:
                        logger.debug(
                            "Rotation estimation %s failed for grid %s angle %s seed %s: %s",
                            method_name,
                            grid_size,
                            true_angle,
                            seed,
                            exc,
                        )
                        est_angle = float("nan")
                        est_success = False

                    record = {
                        "Method": method_name,
                        "Grid Size": grid_size,
                        "Sites": grid_size**2,
                        "Loaded Sites": loaded_sites,
                        "Load Fraction": loaded_sites / float(grid_size**2),
                        "True Angle (deg)": true_angle,
                        "Seed": seed,
                        "Row Spacing (px)": row_spacing,
                        "Col Spacing (px)": col_spacing,
                        "Estimation Success": est_success,
                    }

                    if not est_success or np.isnan(est_angle):
                        record.update(
                            {
                                "Estimated Angle (deg)": float("nan"),
                                "Angle Error (deg)": float("nan"),
                                "Abs Angle Error (deg)": float("nan"),
                                "Mean Displacement (px)": float("nan"),
                                "Max Displacement (px)": float("nan"),
                                "P95 Displacement (px)": float("nan"),
                                "Relative Mean Disp (spacing units)": float("nan"),
                                "Relative Max Disp (spacing units)": float("nan"),
                                "Theoretical Edge Disp (px)": float("nan"),
                                "Assignment Recall": float("nan"),
                                "Assignment Precision": float("nan"),
                                "Assignment Exact Match": False,
                                "Baseline Recall": baseline_stats["recall"],
                                "Baseline Precision": baseline_stats["precision"],
                                "Baseline Exact Match": baseline_stats["exact_match"],
                                "Recall Drop": float("nan"),
                                "Feasible >=99%": False,
                            }
                        )
                        records.append(record)
                        continue

                    corrected = inverse_rotate_centroids(
                        rotated, image_shape=image_shape, angle_deg=est_angle
                    )
                    disp = (
                        np.linalg.norm(corrected - gt_aligned, axis=1)
                        if len(corrected)
                        else np.array([], dtype=float)
                    )
                    binary_pred = fit_grid_and_assign(
                        corrected, (grid_size, grid_size), image_shape=image_shape
                    )
                    stats = _compute_assignment_metrics(binary_pred, true_binary)
                    angle_error = _angle_error_deg(est_angle, true_angle)
                    abs_angle_error = abs(angle_error)
                    angle_error_rad = np.deg2rad(abs_angle_error)
                    theoretical_edge_disp = (
                        float(radii.max() * angle_error_rad) if radii.size else 0.0
                    )
                    mean_disp = float(np.mean(disp)) if disp.size else 0.0
                    max_disp = float(np.max(disp)) if disp.size else 0.0
                    p95_disp = float(np.percentile(disp, 95)) if disp.size else 0.0
                    relative_mean_disp = (
                        mean_disp / avg_spacing if avg_spacing else np.nan
                    )
                    relative_max_disp = (
                        max_disp / avg_spacing if avg_spacing else np.nan
                    )

                    record.update(
                        {
                            "Estimated Angle (deg)": est_angle,
                            "Angle Error (deg)": angle_error,
                            "Abs Angle Error (deg)": abs_angle_error,
                            "Mean Displacement (px)": mean_disp,
                            "Max Displacement (px)": max_disp,
                            "P95 Displacement (px)": p95_disp,
                            "Relative Mean Disp (spacing units)": relative_mean_disp,
                            "Relative Max Disp (spacing units)": relative_max_disp,
                            "Theoretical Edge Disp (px)": theoretical_edge_disp,
                            "Mean Radius (px)": (
                                float(radii.mean()) if radii.size else 0.0
                            ),
                            "Edge Radius (px)": (
                                float(radii.max()) if radii.size else 0.0
                            ),
                            "Assignment Recall": stats["recall"],
                            "Assignment Precision": stats["precision"],
                            "Assignment Exact Match": stats["exact_match"],
                            "Baseline Recall": baseline_stats["recall"],
                            "Baseline Precision": baseline_stats["precision"],
                            "Baseline Exact Match": baseline_stats["exact_match"],
                            "Recall Drop": stats["recall"] - baseline_stats["recall"],
                            "Feasible >=99%": stats["recall"] >= 0.99,
                        }
                    )
                    records.append(record)

    if not records:
        logger.warning(
            "Rotation error benchmark produced no records; nothing to write."
        )
        return

    df = pd.DataFrame.from_records(records)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)

    feasibility_rate = df[df["Estimation Success"]]["Feasible >=99%"].mean()
    logger.info(
        "Saved rotation error benchmark (%s rows) to %s. >=99%% feasibility rate: %.3f",
        len(df),
        output_csv,
        feasibility_rate if not np.isnan(feasibility_rate) else float("nan"),
    )


def benchmark_time_estimation_techniques(
    logger: logging.Logger,
    n_repeats: int = 3,
    grid_size: int = 9,
    image_shape: Tuple[int, int] = (640, 1280),
    angles: Optional[List[float]] = None,
    output_dir: str = "figs/benchmark_pipeline",
) -> None:
    """
    Benchmark the full pipeline: initial extraction on the rotated image,
    timing of each rotation estimation method, timing to rotate the image
    using each method's estimated angle, and timing to run extraction on the
    corrected image. Does not create intermediate plots. Saves per-run
    timings to CSV and a single estimation timing plot.

    Outputs
    -------
    - figs/benchmark_pipeline/benchmark_pipeline_steps.csv
        raw times per run (extraction, rotation, post-extraction) aggregated
    - figs/benchmark_pipeline/benchmark_estimation_times.csv
        per-run per-method estimation times
    - figs/benchmark_pipeline/estimation_timing.svg
        bar plot comparing mean+std of estimation methods
    """

    os.makedirs(output_dir, exist_ok=True)

    if angles is None:
        angles = [-45, -30, -5, 0, 5, 30, 45]

    # The estimation methods to evaluate (same handling as elsewhere).
    estimation_methods = [
        estimate_grid_rotation_diffs,
        estimate_grid_rotation_pca,
        estimate_grid_rotation_pair_diff,
        estimate_grid_rotation_diff_pca,
        estimate_grid_rotation_vectorize,
        estimate_grid_rotation_fit_rect,
        estimate_grid_rotation_fourier_img,
        estimate_grid_rotation_fourier,
    ]

    pipeline_records: List[dict] = []
    estimation_records: List[dict] = []

    total_iterations = len(angles) * n_repeats
    iter_idx = 0

    benchmark_folder_cor = os.path.join(output_dir, "cor_images")
    os.makedirs(benchmark_folder_cor, exist_ok=True)
    benchmark_folder_rot = os.path.join(output_dir, "rot_images")
    os.makedirs(benchmark_folder_rot, exist_ok=True)
    benchmark_folder_map = os.path.join(output_dir, "mappings")
    os.makedirs(benchmark_folder_map, exist_ok=True)

    for angle_idx, angle in enumerate(angles):
        for run in range(n_repeats):
            iter_idx += 1
            logger.info(
                f"Benchmark iter {iter_idx}/{total_iterations}: angle={angle}, run={run}"
            )

            suffix = f"bench_{run}_{int(angle)}"
            seed = 1000 + angle_idx * n_repeats + run
            np.random.seed(seed)
            points, true_binary = generate_rot_img(
                image_shape,
                grid_size,
                true_angle=angle,
                suffix=suffix,
                directory=benchmark_folder_rot,
            )
            ideal_points = np.asarray(points, dtype=float).reshape(-1, 2)

            rot_img_path = os.path.join(benchmark_folder_rot, f"{suffix}_rot_image.png")
            img = plt.imread(rot_img_path)

            # One initial extraction on the rotated image (timed once per run)
            blob_params = setup_blob_params(None)
            extractor_blob = BlobDetection(
                shape=(grid_size, grid_size),
                spots=len(points),
                scale=(1, 1),
                logger=logger,
                blob_params=blob_params,
            )

            t0 = time.perf_counter()
            centroids, _ = extractor_blob.extract(rot_img_path)
            t1 = time.perf_counter()
            extraction_time = t1 - t0

            # For each estimation method: time estimation, rotation with its
            # estimated angle, and post-rotation extraction.
            for method in estimation_methods:
                method_name = method.__name__.replace("estimate_grid_rotation_", "")
                # --- estimation timing ---
                est_angle = float("nan")
                est_success = False
                est_t0 = time.perf_counter()
                try:
                    if method == estimate_grid_rotation_vectorize:
                        est_angle = method(
                            centroids=centroids,
                            grid_shape=(grid_size, grid_size),
                            plot=False,
                        )
                    elif method == estimate_grid_rotation_fourier_img:
                        img_gray = img[..., 0] if img.ndim == 3 else img
                        est_angle = method(img=img_gray, plot=False)
                    elif method == estimate_grid_rotation_fourier:
                        est_angle = method(
                            centroids=centroids, image_shape=image_shape, plot=False
                        )
                    else:
                        est_angle = method(centroids=centroids, plot=False)
                    est_success = True
                except Exception as e:
                    logger.warning(f"Estimation {method_name} failed: {e}")
                est_t1 = time.perf_counter()
                est_time = est_t1 - est_t0

                estimation_records.append(
                    {
                        "Run": run,
                        "Angle": angle,
                        "Method": method_name,
                        "Estimation Time (s)": est_time,
                        "Estimation Success": est_success,
                        "Estimated Angle (deg)": est_angle,
                    }
                )

                # --- instead of rotating the entire image and re-running the
                # expensive extractor, just inverse-rotate the detected
                # centroids back to the original coordinate frame. This is
                # much faster and avoids I/O. Record the transform timing.
                rot_time = float("nan")
                post_ext_success = False
                centroid_transform_error = float("nan")
                mean_offset_y = float("nan")
                mean_offset_x = float("nan")
                rms_offset = float("nan")
                if est_success and not np.isnan(est_angle) and len(centroids) > 0:
                    try:
                        rt0 = time.perf_counter()
                        # inverse_rotate_centroids expects centroids in (y,x)
                        centroids_arr = np.asarray(centroids)
                        centroids_corrected = inverse_rotate_centroids(
                            centroids=centroids_arr,
                            image_shape=img.shape,
                            angle_deg=est_angle,
                        )
                        rt1 = time.perf_counter()
                        rot_time = rt1 - rt0

                        matched_centroids = np.empty((0, 2))
                        matched_true = np.empty((0, 2))
                        if centroids_corrected.size and ideal_points.size:
                            cost_matrix = np.linalg.norm(
                                centroids_corrected[:, None, :]
                                - ideal_points[None, :, :],
                                axis=2,
                            )
                            if cost_matrix.size:
                                row_idx, col_idx = linear_sum_assignment(cost_matrix)
                                matched_centroids = centroids_corrected[row_idx]
                                matched_true = ideal_points[col_idx]
                                if len(matched_centroids) > 0:
                                    residuals = matched_centroids - matched_true
                                    centroid_transform_error = float(
                                        np.mean(np.linalg.norm(residuals, axis=1))
                                    )
                                    mean_offset = np.nanmean(residuals, axis=0)
                                    mean_offset_y, mean_offset_x = float(
                                        mean_offset[0]
                                    ), float(mean_offset[1])
                                    rms_offset = float(
                                        np.sqrt(
                                            np.nanmean(np.sum(residuals**2, axis=1))
                                        )
                                    )

                        # Save a small mapping visualization (corrected centroids vs ideal grid)
                        try:
                            plt.figure(figsize=(6, 6))
                            if ideal_points.size:
                                plt.scatter(
                                    ideal_points[:, 1],
                                    ideal_points[:, 0],
                                    c="lightgray",
                                    s=30,
                                    label="Ideal grid (present)",
                                )
                            if len(centroids_corrected):
                                plt.scatter(
                                    centroids_corrected[:, 1],
                                    centroids_corrected[:, 0],
                                    c="tab:blue",
                                    s=20,
                                    label="Corrected centroids",
                                )
                            for (y0, x0), (y1, x1) in zip(
                                matched_centroids, matched_true
                            ):
                                plt.plot([x0, x1], [y0, y1], c="gray", lw=0.5)

                            plt.gca().invert_yaxis()
                            # plt.title(f"Mapping: {method_name} run={run} angle={angle}")
                            plt.legend(loc="upper right")
                            plt.tight_layout()
                            map_path = os.path.join(
                                benchmark_folder_map,
                                f"mapping_{method_name}_{suffix}.svg",
                            )
                            plt.savefig(map_path, format="svg", dpi=200)
                            plt.close()
                        except Exception:
                            pass

                        post_ext_success = np.array_equal(
                            true_binary,
                            fit_grid_and_assign(
                                centroids_corrected,
                                (grid_size, grid_size),
                                img.shape[:2],
                            )[0],
                        )
                    except Exception as e:
                        logger.warning(
                            f"Centroid inverse-rotation failed for {method_name}: {e}"
                        )

                pipeline_records.append(
                    {
                        "Run": run,
                        "Angle": angle,
                        "Method": method_name,
                        "Initial Extraction Time (s)": extraction_time,
                        "Centroid Transform Time (s)": rot_time,
                        "Centroid Transform Success": post_ext_success,
                        "Centroid Transform Error (px)": centroid_transform_error,
                        "Centroid Offset Mean Y (px)": mean_offset_y,
                        "Centroid Offset Mean X (px)": mean_offset_x,
                        "Centroid Offset RMS (px)": rms_offset,
                    }
                )

    # Persist raw CSVs
    est_df = pd.DataFrame(estimation_records)
    pipe_df = pd.DataFrame(pipeline_records)

    est_csv = os.path.join(output_dir, "benchmark_estimation_times.csv")
    pipe_csv = os.path.join(output_dir, "benchmark_pipeline_steps.csv")
    est_df.to_csv(est_csv, index=False)
    pipe_df.to_csv(pipe_csv, index=False)

    logger.info(f"Wrote estimation times to {est_csv}")
    logger.info(f"Wrote pipeline step times to {pipe_csv}")

    # Produce a concise estimation timing plot (mean +/- sd)
    try:
        grouping = (
            est_df.groupby("Method")["Estimation Time (s)"]
            .agg(["mean", "std"])
            .reset_index()
        )
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=est_df,
            x="Method",
            y="Estimation Time (s)",
            errorbar="sd",
            palette="Blues",
        )
        # ax.set_title("Estimation method timing (mean ± sd) — log scale")
        # add finer y grid lines for the log scale
        ax.yaxis.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_yscale("log")
        plt.ylabel("Estimation Time (s) [log scale]")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "estimation_timing_log.svg")
        plt.savefig(plot_path, format="svg", dpi=200)
        plt.close()
        logger.info(f"Saved estimation timing plot to {plot_path}")
    except Exception as e:
        logger.exception(f"Failed to create estimation timing plot: {e}")

    # Summarize pipeline step timings (mean/std) for the user
    summary = pipe_df.groupby("Method")[
        [
            "Initial Extraction Time (s)",
            "Centroid Transform Time (s)",
            "Centroid Transform Error (px)",
        ]
    ].agg(["mean", "std"])
    summary_csv = os.path.join(output_dir, "pipeline_steps_summary.csv")
    try:
        summary.to_csv(summary_csv)
        logger.info(f"Wrote pipeline summary (mean/std) to {summary_csv}")
    except Exception:
        logger.warning("Could not write pipeline summary CSV")


def benchmark_time_full_extraction_pipeline(
    logger: logging.Logger,
    angles: Optional[List[float]] = None,
    n_repeats: int = 10,
    grid_size: int = 20,
    image_shape: Tuple[int, int] = (400, 400),
    output_csv: str = "data/benchmark_pipeline/full_extraction_times.csv",
) -> None:
    """
    Measure the end-to-end extraction pipeline time (read image -> extract ->
    estimate rotation (PCA or fit_rect) -> inverse-rotate centroids -> assign grid).

    Exclude image generation (assumes images named like "bench_{run}_{angle}_rot_image.png"
    exist under `figs/benchmark_pipeline/rot_images`). For angles with |angle| <= 10°
    also measure a direct mode that skips rotation estimation and immediately tries
    to infer the grid from raw centroids.
    """
    if angles is None:
        angles = [-10, -7, -5, -2, 0, 2, 5, 7, 10]

    modes = ["pca", "fit_rect", "direct"]
    records: List[dict] = []

    rot_folder = os.path.join("figs/benchmark_pipeline", "rot_images")

    np.random.seed(42)

    for angle in angles:
        for run in range(n_repeats):
            suffix = f"bench_{run}_{int(angle)}"
            _, true_binary = generate_rot_img(
                image_shape,
                grid_size,
                true_angle=angle,
                suffix=suffix,
                directory=rot_folder,
            )
            img_path = os.path.join(rot_folder, f"{suffix}_rot_image.png")
            logger.info(f"Processing angle={angle}, run={run}, image={img_path}")

            # time reading the image
            t0 = time.perf_counter()
            img = plt.imread(img_path)
            t1 = time.perf_counter()
            read_time = t1 - t0

            # time centroid extraction from the rotated image
            blob_params = setup_blob_params(None)
            # approx. spots = 0.6 * grid_size ** 2
            extractor_blob = BlobDetection(
                shape=(grid_size, grid_size),
                scale=(1, 1),
                logger=logger,
                blob_params=blob_params,
            )

            t0 = time.perf_counter()
            centroids, _ = extractor_blob.extract(img_path)
            t1 = time.perf_counter()
            extraction_time = t1 - t0

            for mode in modes:
                # skip direct mode for large angles
                if mode == "direct" and abs(angle) > 10:
                    continue

                est_angle = float("nan")
                est_time = float("nan")
                transform_time = float("nan")
                assign_time = float("nan")
                success = False

                full_start = time.perf_counter()

                if mode == "direct":
                    # directly try to fit grid from centroids (no rotation)
                    try:
                        assign_t0 = time.perf_counter()
                        binary_matrix = fit_grid_and_assign(
                            np.asarray(centroids),
                            (grid_size, grid_size),
                            image_shape=img.shape[:2],
                        )
                        assign_t1 = time.perf_counter()
                        assign_time = assign_t1 - assign_t0
                    except Exception as e:
                        logger.warning(f"Direct assign failed: {e}")
                else:
                    if mode == "pca":
                        est_t0 = time.perf_counter()
                        est_angle = estimate_grid_rotation_pca(centroids, plot=False)
                    elif mode == "fit_rect":
                        est_t0 = time.perf_counter()
                        est_angle = estimate_grid_rotation_fit_rect(
                            centroids, plot=False
                        )
                    est_t1 = time.perf_counter()
                    est_time = est_t1 - est_t0

                    tr_t0 = time.perf_counter()
                    centroids_corrected = inverse_rotate_centroids(
                        np.asarray(centroids),
                        image_shape=img.shape,
                        angle_deg=est_angle,
                    )
                    tr_t1 = time.perf_counter()
                    transform_time = tr_t1 - tr_t0

                    assign_t0 = time.perf_counter()
                    binary_matrix = fit_grid_and_assign(
                        centroids_corrected,
                        (grid_size, grid_size),
                        image_shape=img.shape[:2],
                    )
                    assign_t1 = time.perf_counter()
                    assign_time = assign_t1 - assign_t0

                # Visualization/debugging: plot binary_matrix vs true_binary
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.title("Assigned Binary")
                plt.imshow(binary_matrix, cmap="gray")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.title("True Binary")
                plt.imshow(true_binary, cmap="gray")
                plt.axis("off")

                plt.tight_layout()
                plt.savefig(
                    f"figs/benchmark_pipeline/rot_images/bench_{angle}_comparison.png"
                )

                # check correctness

                success = np.array_equal(binary_matrix, true_binary)
                logger.info(
                    f"Pipeline correct: {success} (mode={mode}, angle={angle}, run={run})"
                )
                if not success and False:
                    incorrect_coords = np.argwhere(binary_matrix != true_binary)
                    logger.debug(f"Incorrect blob coordinates: {incorrect_coords}")
                    for coord in incorrect_coords:
                        logger.debug(
                            f"At {coord}, expected {true_binary[coord[0]][coord[1]]} but got {binary_matrix[coord[0]][coord[1]]}"
                        )

                full_end = time.perf_counter()
                full_time = full_end - full_start

                records.append(
                    {
                        "Run": run,
                        "Angle": angle,
                        "Mode": mode,
                        "Image Read Time (s)": read_time,
                        "Extraction Time (s)": extraction_time,
                        "Estimation Time (s)": est_time,
                        "Transform Time (s)": transform_time,
                        "Assign Time (s)": assign_time,
                        "Full Pipeline Time (s)": full_time,
                        "Success": success,
                    }
                )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote full extraction timing to {output_csv}")

    # Calculate and log mean/std for each mode
    if not df.empty:
        summary = df.groupby("Mode")[
            [
                "Extraction Time (s)",
                "Estimation Time (s)",
                "Transform Time (s)",
                "Assign Time (s)",
                "Full Pipeline Time (s)",
            ]
        ].agg(["mean", "std"])

        logger.info("\n--- Full Extraction Pipeline Timing Summary ---")
        logger.info(summary.to_string())

        summary_csv = os.path.join(
            os.path.dirname(output_csv), "full_extraction_summary.csv"
        )
        summary.to_csv(summary_csv)
        logger.info(f"Wrote summary to {summary_csv}")


def test_final_extraction_pipeline(logger):
    """
    Test the full extraction pipeline on a single image with a moderate rotation.

    Generates plots of each step for visual inspection.
    """
    angle = 15
    grid_size = 9
    image_shape = (640, 1280)
    suffix = "test_final"
    np.random.seed(42)

    _, true_binary = generate_rot_img(
        image_shape,
        grid_size,
        true_angle=angle,
        suffix=suffix,
        directory="figs/test_final",
    )
    rot_img_path = os.path.join("figs/test_final", f"{suffix}_rot_image.png")
    img = plt.imread(rot_img_path)

    blob_params = setup_blob_params(None)
    extractor_blob = BlobDetection(
        shape=(grid_size, grid_size),
        scale=(1, 1),
        logger=logger,
        blob_params=blob_params,
    )

    centroids, _ = extractor_blob.extract(rot_img_path)
    logger.info(f"Extracted {len(centroids)} centroids from rotated image.")

    # Estimate rotation using PCA and fit_rect (compare results)
    est_angle_fit_rect = estimate_grid_rotation_fit_rect(centroids, plot=True)
    logger.info(f"Estimated angles: Fit Rect={est_angle_fit_rect:.2f}")

    # Inverse-rotate centroids using PCA estimate
    centroids_corrected = inverse_rotate_centroids(
        np.asarray(centroids), image_shape=img.shape, angle_deg=est_angle_fit_rect
    )
    binary_centroids = fit_grid_and_assign(
        np.asarray(centroids_corrected),
        (grid_size, grid_size),
        image_shape=img.shape[:2],
    )

    assert np.array_equal(
        binary_centroids, true_binary
    ), "Final binary from centroids does not match true binary."
    logger.info(
        "Final extraction pipeline test passed: binary from centroids matches true binary."
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark and test grid rotation estimation methods."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the benchmark of estimation methods.",
    )
    parser.add_argument(
        "--feasibility",
        action="store_true",
        help="Run feasibility tests of estimation methods.",
    )
    parser.add_argument(
        "--time_pipeline",
        action="store_true",
        help="Benchmark the full extraction pipeline timing.",
    )
    parser.add_argument(
        "--time_full_extraction",
        action="store_true",
        help="Benchmark full extraction pipeline with different modes.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize results of estimation techniques on a test image.",
    )
    parser.add_argument(
        "--test_final",
        action="store_true",
        help="Test the final extraction pipeline on a single image.",
    )
    parser.add_argument(
        "--rotation_error_benchmark",
        action="store_true",
        help="Benchmark PCA vs. rectangle fitting impact on grid assignment feasibility.",
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="Generate plots from benchmark results.",
    )
    parser.add_argument(
        "--feasibility_csv",
        type=str,
        default="data/benchmark_pipeline/feasibility_results.csv",
        help="Output CSV file for feasibility results.",
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="data/benchmark_pipeline",
        help="Directory for benchmark pipeline outputs.",
    )
    parser.add_argument(
        "--rotation_error_csv",
        type=str,
        default="data/benchmark_pipeline/rotation_error_impact.csv",
        help="Output CSV for rotation error feasibility benchmark.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename="test_imaging.log", encoding="utf-8", level=logging.INFO
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Starting main pipeline...")

    test_grid_extraction(logger)

    if args.benchmark:
        test_estimation_and_extraction(logger)

        algorithms = [
            estimate_grid_rotation_diffs,
            estimate_grid_rotation_pca,
            estimate_grid_rotation_pair_diff,
            estimate_grid_rotation_diff_pca,
            estimate_grid_rotation_vectorize,
            estimate_grid_rotation_fit_rect,
            estimate_grid_rotation_fourier_img,
            estimate_grid_rotation_fourier,
        ]

        for algo in algorithms:
            estimation_method_test(logger, method=algo, path=args.benchmark_dir)

        if args.make_plots:
            make_plots(
                source_csv=args.benchmark_dir + "/benchmark_estimation.csv",
                save_dir=args.benchmark_dir,
            )

    if args.feasibility:
        test_estimation_feasibility(logger, output_csv=args.feasibility_csv)

    if args.time_pipeline:
        benchmark_time_estimation_techniques(logger, output_dir=args.benchmark_dir)

    if args.time_full_extraction:
        benchmark_time_full_extraction_pipeline(
            logger,
            output_csv=os.path.join(args.benchmark_dir, "full_extraction_times.csv"),
        )

    if args.rotation_error_benchmark:
        benchmark_rotation_error_feasibility(logger, output_csv=args.rotation_error_csv)

    if args.visualize:
        methods = [
            estimate_grid_rotation_diffs,
            estimate_grid_rotation_pca,
            estimate_grid_rotation_pair_diff,
            estimate_grid_rotation_diff_pca,
            estimate_grid_rotation_vectorize,
            estimate_grid_rotation_fit_rect,
            estimate_grid_rotation_fourier_img,
            estimate_grid_rotation_fourier,
        ]
        visualize_result_techniques(methods, true_angle=15)

    if args.test_final:
        test_final_extraction_pipeline(logger)


if __name__ == "__main__":
    main()

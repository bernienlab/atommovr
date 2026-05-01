from __future__ import annotations

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Polygon
from PIL import Image
from sklearn.cluster import KMeans

from .geometry import rotate_points_ccw

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def _wrap_angle_deg(angle_deg: float) -> float:
    """Wrap arbitrary angle to [-90, 90)."""
    return ((angle_deg + 90.0) % 180.0) - 90.0


def _pca_axis_angle(
    centroids: np.ndarray,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return PCA-based angle (degrees), axis, and singular values."""
    centroids = np.asarray(centroids)
    if len(centroids) < 2:
        return 0.0, None, None
    mean = centroids.mean(axis=0)
    centered = centroids - mean
    _, singular_values, Vt = np.linalg.svd(centered)
    main_axis = Vt[0]
    raw = np.degrees(np.arctan2(main_axis[1], main_axis[0])) - 90.0
    return _wrap_angle_deg(raw), main_axis, singular_values


def fit_grid_and_assign(
    centroids: np.ndarray,
    grid_shape: Tuple[int, int],
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    centroids = np.asarray(centroids)
    R, C = grid_shape
    N = len(centroids)
    if N == 0:
        return np.zeros((R, C), dtype=int)
    if N < 3:
        if image_shape is not None:
            img_h, img_w = image_shape[:2]
        else:
            img_h = centroids[:, 0].max() - centroids[:, 0].min()
            img_w = centroids[:, 1].max() - centroids[:, 1].min()
        min_y, min_x = centroids.min(axis=0)
        row_spacing = img_h / max(R, 1)
        col_spacing = img_w / max(C, 1)
        row_idx = np.round((centroids[:, 0] - min_y) / row_spacing).astype(int)
        col_idx = np.round((centroids[:, 1] - min_x) / col_spacing).astype(int)
        row_idx = np.clip(row_idx, 0, R - 1)
        col_idx = np.clip(col_idx, 0, C - 1)
    else:
        mean = centroids.mean(axis=0)
        centered = centroids - mean
        row_coords = centered[:, 0]
        col_coords = centered[:, 1]
        row_bins = np.linspace(row_coords.min(), row_coords.max(), R + 1)
        col_bins = np.linspace(col_coords.min(), col_coords.max(), C + 1)
        row_idx = np.digitize(row_coords, row_bins) - 1
        col_idx = np.digitize(col_coords, col_bins) - 1
        row_idx = np.clip(row_idx, 0, R - 1)
        col_idx = np.clip(col_idx, 0, C - 1)

    binary = np.zeros((R, C), dtype=int)
    for r, c in zip(row_idx, col_idx, strict=True):
        binary[r, c] = 1
    return binary


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    # Prefer OpenCV when available for speed; otherwise use PIL fallback.
    if cv2 is not None:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        return rotated

    # PIL fallback: convert to Image, rotate, convert back.
    pil_mode = None
    if image.ndim == 2:
        pil_mode = "L"
    elif image.shape[2] == 3:
        pil_mode = "RGB"
    elif image.shape[2] == 4:
        pil_mode = "RGBA"
    else:
        pil_mode = None

    if pil_mode is None:
        # Generic fallback: rotate via scipy/numpy affine (simple nearest-neighbor)
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        h, w = image.shape[:2]
        center_x, center_y = w / 2.0, h / 2.0
        Y, X = np.indices((h, w))
        Xc = X - center_x
        Yc = Y - center_y
        Xr = Xc * cos_t - Yc * sin_t + center_x
        Yr = Xc * sin_t + Yc * cos_t + center_y
        Xr = np.clip(np.round(Xr).astype(int), 0, w - 1)
        Yr = np.clip(np.round(Yr).astype(int), 0, h - 1)
        return image[Yr, Xr]

    pil_img = Image.fromarray(image.astype("uint8"), mode=pil_mode)
    rotated_pil = pil_img.rotate(angle_deg, resample=Image.BILINEAR, expand=False)
    return np.array(rotated_pil)


def estimate_grid_rotation_pca(
    centroids: np.ndarray,
    plot: bool = False,
    return_diagnostics: bool = False,
) -> float | Tuple[float, dict]:
    centroids = np.asarray(centroids)
    if len(centroids) < 2:
        return (0.0, {"singular_values": None}) if return_diagnostics else 0.0
    angle_deg, axis, singular_values = _pca_axis_angle(centroids)
    main_axis = axis
    mean = centroids.mean(axis=0)
    if plot:
        os.makedirs("figs/estimation_tech/", exist_ok=True)
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(centroids[:, 1], centroids[:, 0], alpha=0.6, s=20, color="#2b8cbe")
        ax.scatter(mean[1], mean[0], c="red", marker="x", s=60, label="Mean")
        scale = np.max(np.ptp(centroids, axis=0)) * 0.6 if centroids.size else 50
        ax.plot(
            [mean[1] - scale * main_axis[0], mean[1] + scale * main_axis[0]],
            [mean[0] - scale * main_axis[1], mean[0] + scale * main_axis[1]],
            "r-",
            lw=2,
            label=f"PCA axis ({angle_deg:.2f}°)",
        )
        ax.set_xlabel("x (col)")
        ax.set_ylabel("y (row)")
        ax.set_title("PCA of centroids")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig("figs/estimation_tech/pca_direction.svg", bbox_inches="tight")
        plt.close()
    if return_diagnostics:
        diag = {
            "singular_values": singular_values[:2],
            "principal_axis": main_axis,
        }
        return angle_deg, diag
    return angle_deg


def estimate_grid_rotation_vectorize(
    centroids: np.ndarray, grid_shape: Tuple[int, int], plot: bool = False
) -> float:
    centroids = np.asarray(centroids)
    if len(centroids) < 2:
        return 0.0
    sorted_idx = np.lexsort((centroids[:, 1], centroids[:, 0]))
    centroids_sorted = centroids[sorted_idx]
    rows, cols = grid_shape
    padded = np.full((rows * cols, 2), np.nan)
    padded[: len(centroids_sorted)] = centroids_sorted
    grid_like = padded.reshape(rows, cols, 2)
    horiz_vecs, vert_vecs = [], []
    for r in range(rows):
        row = grid_like[r, :, :]
        for c in range(cols - 1):
            if not np.any(np.isnan(row[c])) and not np.any(np.isnan(row[c + 1])):
                horiz_vecs.append(row[c + 1] - row[c])
    for c in range(cols):
        col = grid_like[:, c, :]
        for r in range(rows - 1):
            if not np.any(np.isnan(col[r])) and not np.any(np.isnan(col[r + 1])):
                vert_vecs.append(col[r + 1] - col[r])
    horiz_vecs = np.array(horiz_vecs) if len(horiz_vecs) else np.zeros((0, 2))
    h_mean = np.nanmean(horiz_vecs, axis=0) if horiz_vecs.size else np.array([1.0, 0.0])
    main_axis = (
        h_mean / np.linalg.norm(h_mean) if np.linalg.norm(h_mean) != 0 else h_mean
    )
    angle = np.arctan2(main_axis[1], main_axis[0])
    angle_deg = np.degrees(angle)
    angle_deg = (angle_deg) % 180 - 90
    if plot:
        os.makedirs("figs/estimation_tech/", exist_ok=True)
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(centroids[:, 1], centroids[:, 0], alpha=0.6, s=20, color="#2b8cbe")
        mean = centroids.mean(axis=0)
        ax.scatter(mean[1], mean[0], c="red", marker="x", s=60, label="Mean")
        scale = max(np.ptp(centroids, axis=0)) * 0.15 if centroids.size else 50
        for v in horiz_vecs:
            ax.arrow(
                mean[1],
                mean[0],
                v[1] * scale / max(np.linalg.norm(v), 1e-8),
                v[0] * scale / max(np.linalg.norm(v), 1e-8),
                color="#4daf4a",
                alpha=0.15,
                head_width=1,
                length_includes_head=True,
            )
        for v in vert_vecs:
            ax.arrow(
                mean[1],
                mean[0],
                v[1] * scale / max(np.linalg.norm(v), 1e-8),
                v[0] * scale / max(np.linalg.norm(v), 1e-8),
                color="#984ea3",
                alpha=0.15,
                head_width=1,
                length_includes_head=True,
            )
        ax.plot(
            [mean[1] - scale * main_axis[0], mean[1] + scale * main_axis[0]],
            [mean[0] - scale * main_axis[1], mean[0] + scale * main_axis[1]],
            color="red",
            lw=2,
            label=f"Main axis ({angle_deg:.2f}°)",
        )
        ax.set_xlabel("x (col)")
        ax.set_ylabel("y (row)")
        ax.set_title("Vectorized row/col differences")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig("figs/estimation_tech/vectorize.svg", bbox_inches="tight")
        plt.close()
    return angle_deg


def estimate_grid_rotation_diff_pca(centroids: np.ndarray, plot: bool = False) -> float:
    centroids = np.asarray(centroids)
    N = len(centroids)
    if N < 2:
        return 0.0
    max_pairs = 20000
    try:
        from scipy.spatial import cKDTree

        k = min(8, max(1, N - 1))
        tree = cKDTree(centroids)
        dists, idxs = tree.query(centroids, k=k + 1, n_jobs=-1)
        diffs_list = []
        for i in range(N):
            neigh = idxs[i, 1:]
            diffs_list.append(centroids[neigh] - centroids[i])
        diffs = np.vstack(diffs_list)
    except Exception:
        if N * (N - 1) // 2 <= max_pairs:
            ii, jj = np.triu_indices(N, k=1)
            diffs = centroids[jj] - centroids[ii]
        else:
            rng = np.random.default_rng(0)
            num_pairs = max_pairs
            i_idx = rng.integers(0, N, size=num_pairs)
            j_idx = rng.integers(0, N, size=num_pairs)
            mask = i_idx != j_idx
            i_idx = i_idx[mask]
            j_idx = j_idx[mask]
            if len(i_idx) > max_pairs:
                sel = rng.choice(len(i_idx), size=max_pairs, replace=False)
                i_idx = i_idx[sel]
                j_idx = j_idx[sel]
            diffs = centroids[j_idx] - centroids[i_idx]
    norms = np.linalg.norm(diffs, axis=1)
    valid = norms > 1e-8
    if not np.any(valid):
        return 0.0
    diffs = diffs[valid]
    mean = diffs.mean(axis=0)
    centered = diffs - mean
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main_axis = eigvecs[:, np.argmax(eigvals)]
    else:
        main_axis = Vt[0]
    angle_rad = np.arctan2(main_axis[0], main_axis[1])
    angle_deg = np.degrees(angle_rad)
    angle_deg = (angle_deg + 90) % 180 - 90
    if plot:
        try:
            os.makedirs("figs/estimation_tech/", exist_ok=True)
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                centroids[:, 1], centroids[:, 0], alpha=0.6, s=20, color="#2b8cbe"
            )
            subsample = np.random.choice(
                len(diffs), size=min(200, len(diffs)), replace=False
            )
            origin_x, origin_y = 0, 0
            for d in diffs[subsample]:
                ax.arrow(
                    origin_x,
                    origin_y,
                    d[1],
                    d[0],
                    color="gray",
                    alpha=0.08,
                    head_width=0.5,
                    length_includes_head=True,
                )
            scale = max(np.ptp(centroids, axis=0)) * 0.6 if centroids.size else 50
            ax.plot(
                [0, scale * main_axis[1]],
                [0, scale * main_axis[0]],
                "r-",
                lw=2,
                label=f"Main axis ({angle_deg:.2f}°)",
            )
            ax.set_xlabel("Δx")
            ax.set_ylabel("Δy")
            ax.set_title("PCA on sampled pairwise differences")
            ax.set_aspect("equal")
            ax.legend(loc="best", fontsize="small")
            plt.tight_layout()
            plt.savefig(
                "figs/estimation_tech/pca_direction_diffs.svg", bbox_inches="tight"
            )
            plt.close()
        except Exception:
            pass
    return -angle_deg


def estimate_grid_rotation_diffs(centroids: np.ndarray, plot: bool = False) -> float:
    centroids = np.asarray(centroids)
    diffs = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = centroids[j] - centroids[i]
            if np.linalg.norm(d) < 100:
                diffs.append(d / np.linalg.norm(d))
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return 0.0
    kmeans = KMeans(n_clusters=2, n_init=10).fit(diffs)
    main_dir = kmeans.cluster_centers_[0]
    angle_rad = np.arctan2(main_dir[0], main_dir[1])
    angle_deg = np.degrees(-angle_rad)
    angle_deg = (angle_deg) % 180 - 90
    if plot:
        os.makedirs("figs/estimation_tech/", exist_ok=True)
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            diffs[:, 1],
            diffs[:, 0],
            alpha=0.25,
            c=kmeans.labels_,
            cmap="coolwarm",
            s=20,
            label="Diffs",
        )
        for _idx, center in enumerate(kmeans.cluster_centers_):
            ax.arrow(
                0,
                0,
                center[1],
                center[0],
                color="black",
                width=0.005,
                head_width=0.05,
                length_includes_head=True,
            )
            ax.scatter(center[1], center[0], c="black", marker="x", s=80)
        ax.arrow(
            0,
            0,
            main_dir[1],
            main_dir[0],
            color="red",
            width=0.01,
            head_width=0.08,
            length_includes_head=True,
            label="Main direction",
        )
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Δx (normalized)")
        ax.set_ylabel("Δy (normalized)")
        ax.set_title("KMeans on normalized pairwise differences")
        ax.legend(loc="upper right", fontsize="small")
        plt.tight_layout()
        plt.savefig("figs/estimation_tech/kmeans_diffs.svg", bbox_inches="tight")
        plt.close()
    return angle_deg


def estimate_grid_rotation_pair_diff(
    centroids: np.ndarray, plot: bool = False
) -> float:
    centroids = np.asarray(centroids, dtype=float)
    if centroids.size == 0:
        return 0.0
    diffs = centroids[:, None, :] - centroids[None, :, :]
    diffs = diffs.reshape(-1, 2)
    norms = np.linalg.norm(diffs, axis=1)
    valid = norms > 1e-8
    if not np.any(valid):
        return 0.0
    diffs = diffs[valid]
    angles = np.arctan2(diffs[:, 0], diffs[:, 1])
    # Double-angle trick collapses the 90° ambiguity between lattice axes.
    sin2 = np.sin(2.0 * angles)
    cos2 = np.cos(2.0 * angles)
    mean_sin2 = np.mean(sin2)
    mean_cos2 = np.mean(cos2)
    angle_rad = 0.5 * np.arctan2(mean_sin2, mean_cos2)
    angle_deg = -_wrap_angle_deg(np.degrees(angle_rad))
    if plot:
        os.makedirs("figs/estimation_tech/", exist_ok=True)
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(centroids[:, 1], centroids[:, 0], alpha=0.6, s=20, color="#2b8cbe")
        ax1.set_title("Centroids")
        ax1.invert_yaxis()
        ax1.set_xlabel("x (col)")
        ax1.set_ylabel("y (row)")
        ax1.set_aspect("equal")
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        ax2.hist(angles, bins=180, color="#fdae61", edgecolor="k")
        ax2.axvline(angle_rad, color="red", lw=2, label=f"{angle_deg:.2f}°")
        ax2.set_title("Pairwise angle histogram")
        ax2.legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig("figs/estimation_tech/pair_diff.svg", bbox_inches="tight")
        plt.close()
    return angle_deg


def estimate_grid_rotation_fit_rect(centroids: np.ndarray, plot: bool = False) -> float:
    centroids = np.asarray(centroids)
    if len(centroids) < 3:
        return 0.0

    def convex_hull(centroids: np.ndarray) -> np.ndarray:
        pts = np.unique(centroids, axis=0)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        if len(pts) <= 2:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))
        hull = np.array(lower[:-1] + upper[:-1])
        return hull

    def rotate(centroids: np.ndarray, angle: float) -> np.ndarray:
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return centroids @ R.T

    def get_min_abs_angle(rect: np.ndarray) -> float:
        # Get the first edge
        p0 = rect[0]
        p1 = rect[1]
        edge = p1 - p0

        # Calculate angle in degrees
        # centroids are (y, x), so edge is (dy, dx)
        # arctan2(dy, dx) gives angle relative to X-axis (horizontal)
        angle = np.degrees(np.arctan2(edge[0], edge[1]))

        # Map to [-45, 45] to find minimum absolute rotation from an axis
        # This handles the square grid ambiguity by picking the smallest rotation
        min_angle = (angle + 45) % 90 - 45

        return -min_angle

    def min_area_rect(centroids: np.ndarray):
        hull = convex_hull(centroids)
        n = len(hull)
        if n == 0:
            return None
        best = {"area": np.inf, "angle": 0.0, "rect": None}
        for i in range(n):
            p1 = hull[i]
            p2 = hull[(i + 1) % n]
            edge = p2 - p1
            angle = -np.arctan2(edge[1], edge[0])
            rot = rotate(hull, angle)
            min_h, min_w = rot.min(axis=0)
            max_h, max_w = rot.max(axis=0)
            rect_rot = np.array(
                [[min_h, min_w], [max_h, min_w], [max_h, max_w], [min_h, max_w]]
            )
            rect_world = rotate(rect_rot, -angle)
            area = (max_h - min_h) * (max_w - min_w)
            if area < best["area"]:
                angle_deg = get_min_abs_angle(rect_world)
                best = {"area": area, "angle": angle_deg, "rect": rect_world}
        return best

    best = min_area_rect(centroids)
    if best is None:
        return 0.0

    if plot:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(centroids[:, 1], centroids[:, 0], "o", color="#2b8cbe")
        rect_patch = Polygon(
            best["rect"][:, [1, 0]], closed=True, fill=False, color="green", lw=2
        )
        ax.add_patch(rect_patch)
        ax.set_aspect("equal")
        plt.tight_layout()
        os.makedirs("figs/estimation_tech/", exist_ok=True)
        plt.savefig(
            "figs/estimation_tech/min_area_rect_steps.svg", dpi=120, bbox_inches="tight"
        )
        plt.close()
    return best["angle"]


def estimate_grid_rotation_fourier_img(img: np.ndarray, plot: bool = False) -> float:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    h, w = img.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    img_win = img * window
    F = np.abs(np.fft.fftshift(np.fft.fft2(img_win)))
    F = np.log1p(F)
    cy, cx = np.array(F.shape) // 2
    F[cy - 10 : cy + 11, cx - 10 : cx + 11] = 0
    Y, X = np.indices(F.shape)
    angles = np.arctan2(Y - cy, X - cx)
    mags = F.ravel()
    angs = angles.ravel()
    bins = np.linspace(-np.pi / 2, np.pi / 2, 720)
    hist, _ = np.histogram(angs, bins=bins, weights=mags)
    angle_idx = np.argmax(hist)
    angle_rad = (bins[angle_idx] + bins[angle_idx + 1]) / 2
    angle_deg = np.degrees(angle_rad)
    angle_deg = (angle_deg + 90) % 180 - 90
    if plot:
        os.makedirs("figs/estimation_tech/", exist_ok=True)
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img, cmap="Blues")
        axs[0].axis("off")
        axs[0].set_title("Input image")
        axs[1].imshow(F, cmap="hot")
        axs[1].axis("off")
        axs[1].set_title("FFT spectrum (log)")
        axs[2].plot(np.degrees(bins[:-1]), hist, color="#fdae61")
        axs[2].axvline(angle_deg, color="r", linestyle="--", label=f"{angle_deg:.2f}°")
        axs[2].legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(
            "figs/estimation_tech/fft_image_rotation.svg", dpi=200, bbox_inches="tight"
        )
        plt.close()
    return angle_deg


def estimate_grid_rotation_fourier(
    centroids: np.ndarray, image_shape: Tuple[int, int], plot: bool = False
) -> float:
    mask = np.zeros(image_shape, dtype=np.float32)
    for y, x in np.asarray(centroids).astype(int):
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
            mask[y, x] = 1
    F = np.abs(np.fft.fftshift(np.fft.fft2(mask)))
    cy, cx = np.array(F.shape) // 2
    F[cy - 5 : cy + 6, cx - 5 : cx + 6] = 0
    Y, X = np.indices(F.shape)
    angles = -np.arctan2(Y - cy, X - cx)
    mags = F.ravel()
    angs = angles.ravel()
    bins = np.linspace(-np.pi / 2, np.pi / 2, 720)
    hist, _ = np.histogram(angs, bins=bins, weights=mags)
    angle_idx = np.argmax(hist)
    angle_rad = (bins[angle_idx] + bins[angle_idx + 1]) / 2
    angle_deg = np.degrees(angle_rad)
    angle_deg = (angle_deg + 90) % 180 - 90
    if plot:
        os.makedirs("figs/estimation_tech/", exist_ok=True)
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(mask, cmap="Blues")
        axs[0].axis("off")
        axs[0].set_title("Centroid Mask")
        axs[1].imshow(F, cmap="hot")
        axs[1].axis("off")
        axs[1].set_title("FFT Spectrum")
        axs[2].plot(np.degrees(bins[:-1]), hist, color="#fdae61")
        axs[2].axvline(angle_deg, color="b", linestyle="-.", label=f"{angle_deg:.2f}°")
        axs[2].legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(
            "figs/estimation_tech/fft_centroid_rotation.svg",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()
    return angle_deg


def inverse_rotate_centroids(
    centroids: np.ndarray, image_shape: Tuple[int, int], angle_deg: float
) -> np.ndarray:
    """Undo a counter-clockwise rotation of ``angle_deg`` degrees."""
    centroids = np.asarray(centroids, dtype=float)
    if centroids.size == 0:
        return centroids.reshape(-1, 2)
    return rotate_points_ccw(centroids, image_shape, -float(angle_deg))


class Extractor:
    # ------------------------------------------------------------------
    # Portions of the Extractor implementation (this class and related
    # extraction helpers) were adapted from the `slmsuite` project and are
    # included here under the original MIT license. The slmsuite copyright
    # and license must be preserved in any redistributed copies of these
    # portions.
    #
    # Original source:
    #   slmsuite Developers (2021-2025), MIT License
    #   (see included header in this file and project-level LICENSE)
    #
    # Adaptation and integration: Sascha Benz (Ludwig-Maximilians-Universitaet
    # Muenchen / Ryd-Yb tweezer array group). Any modifications to the
    # adapted code are covered by the repository MIT license (see /LICENSE).
    # ------------------------------------------------------------------
    def __init__(
        self,
        shape,
        spots=3,
        threshold=0.1,
        scale=(1.0, 1.0),
        affine_matrix=None,
        logger=None,
    ):
        self.shape = shape
        self.spots = spots
        self.threshold = threshold
        self.scale = scale
        self.affine_matrix = affine_matrix
        self.logger = logger or _get_default_logger()

    def extract(self, image):
        raise NotImplementedError

    @staticmethod
    def centroids_to_binary_grid(centroids, grid_shape):
        centroids = np.array(centroids)
        if len(centroids) == 0:
            return np.zeros(grid_shape, dtype=int)
        mean = centroids.mean(axis=0)
        centered = centroids - mean
        U, S, Vt = np.linalg.svd(centered)
        axes = Vt[:2]
        projected = centered @ axes.T
        n_rows, n_cols = grid_shape
        row_coords = projected[:, 0]
        col_coords = projected[:, 1]
        row_bins = np.linspace(row_coords.min(), row_coords.max(), n_rows + 1)
        col_bins = np.linspace(col_coords.min(), col_coords.max(), n_cols + 1)
        row_idx = np.digitize(row_coords, row_bins) - 1
        col_idx = np.digitize(col_coords, col_bins) - 1
        row_idx = np.clip(row_idx, 0, n_rows - 1)
        col_idx = np.clip(col_idx, 0, n_cols - 1)
        binary_grid = np.zeros(grid_shape, dtype=int)
        for r, c in zip(row_idx, col_idx, strict=True):
            binary_grid[r, c] = 1
        return binary_grid

    def extract_estimate_rotate_and_assign(
        self,
        image: np.ndarray | str,
        grid_shape: Optional[Tuple[int, int]] = None,
        angle_method: Optional[str] = None,
        visualize: bool = False,
        return_details: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, float, int]:
        """
        Convenience pipeline: detect centroids, optionally estimate/rectify rotation,
        and assign to a binary grid.

        Parameters
        ----------
        image : np.ndarray | str
            Input image as a numpy array or file path.
        grid_shape : Tuple[int, int]
            The expected shape of the grid (rows, columns). Defaults to `self.shape`.
        angle_method : Optional[str], optional
            The method to use for angle estimation. Default is None.
        visualize : bool, optional
            Whether to visualize the detected centroids on the image. Default is False.

        Returns
        -------
        np.ndarray | Tuple[np.ndarray, float, int]
            Returns the binary grid by default. If `return_details=True`, returns
            (binary grid, angle in degrees, number of centroids).
        """
        if grid_shape is None:
            grid_shape = tuple(self.shape)
        if isinstance(image, str):
            centroids, img_shape = self.extract(image)
            img_arr = np.array(Image.open(image))
        else:
            centroids, img_shape = self.extract(image)
            img_arr = image

        centroids = np.asarray(centroids)
        angle = 0.0
        if angle_method:
            am = angle_method.lower()
            if am == "pca":
                angle = estimate_grid_rotation_pca(centroids)
            elif am == "vectorize":
                angle = estimate_grid_rotation_vectorize(centroids, grid_shape)
            elif am == "diff_pca":
                angle = estimate_grid_rotation_diff_pca(centroids)
            elif am == "diffs":
                angle = estimate_grid_rotation_diffs(centroids)
            elif am == "pair_diff":
                angle = estimate_grid_rotation_pair_diff(centroids)
            elif am == "fit_rect":
                angle = estimate_grid_rotation_fit_rect(centroids)
            elif am == "fft_img":
                angle = estimate_grid_rotation_fourier_img(img_arr)
            elif am == "fft_mask":
                angle = estimate_grid_rotation_fourier(centroids, img_shape)
            else:
                raise ValueError("Unknown angle_method: %s" % angle_method)
        # optionally rectify image/centroids
        if angle_method in {"fft_img"}:
            rot = rotate_image(img_arr, -angle)
            # Re-extract on rotated image for better assignment
            centroids2, _ = self.extract(rot)
            centroids_use = np.asarray(centroids2)
        else:
            centroids_use = centroids
        binary = fit_grid_and_assign(centroids_use, grid_shape, img_shape)
        angle_deg = float(angle)
        n_centroids = int(len(centroids_use))

        if visualize:
            if isinstance(image, str):
                image_path = image
                img = np.array(Image.open(image_path))
            else:
                img = image
            print(
                f"Centroids shape: {np.array(centroids).shape}, centroids count: {len(centroids)}"
            )
            self.overlay_centroids(
                img, centroids=centroids, save_path="figs/centroids_on_image.png"
            )
        if return_details:
            return binary, angle_deg, n_centroids
        return binary

    @staticmethod
    def overlay_centroids(
        image: np.ndarray,
        centroids: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        Overlay detected centroids on the given image.

        Parameters
        - image: 2D numpy array representing the image.
        - centroids: Nx2 numpy array of (y, x) coordinates of centroids.
        - save_path: Optional path to save the overlaid image. If None, displays the image.
        """
        # if centroids is not None and n_detected > 0:
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        centroids_np = np.asarray(centroids)
        ax.imshow(image, cmap="Blues", origin="upper")
        if centroids_np.size:
            ax.scatter(
                centroids_np[:, 1],
                centroids_np[:, 0],
                s=80,
                facecolors="none",
                edgecolors="red",
                linewidths=1,
            )
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(
                save_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True
            )
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


class BlobDetection(Extractor):
    def __init__(
        self,
        shape,
        spots=None,
        threshold=0.1,
        scale=(1.0, 1.0),
        affine_matrix=None,
        logger=None,
        blob_params=None,
    ):
        super().__init__(shape, spots, threshold, scale, affine_matrix, logger=logger)
        if blob_params is None:
            self.blob_params = cv2.SimpleBlobDetector_Params()
            self.blob_params.filterByColor = True
            self.blob_params.blobColor = 255
            self.blob_params.minThreshold = 80
            self.blob_params.maxThreshold = 255
            self.blob_params.thresholdStep = 20
            self.blob_params.minDistBetweenBlobs = 10
            self.blob_params.minArea = 5
            self.blob_params.maxArea = 1000
            self.blob_params.filterByArea = False
            self.blob_params.filterByCircularity = False
            self.blob_params.filterByConvexity = False
            self.blob_params.filterByInertia = False
        else:
            self.blob_params = blob_params

        self.detector = cv2.SimpleBlobDetector_create(self.blob_params)

    @staticmethod
    def _make_8bit(img):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def extract(self, image: np.ndarray | str) -> Tuple[np.ndarray, Tuple[int, int]]:
        # Accept file path, numpy array, or PIL image
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"cv2.imread failed to read '{image}'")
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)

        H, W = self.shape

        # Convert to grayscale if needed
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        img_8bit = self._make_8bit(np.copy(img_gray))

        # Try OpenCV's circle grid detection first (ordered grid)
        # try:
        ret, centroids = cv2.findCirclesGrid(
            img_8bit, patternSize=(W, H), blobDetector=self.detector
        )
        # except Exception:
        #     ret = False

        # if ret:
        #     self.logger.info("Detected corners using OpenCV findCirclesGrid")
        #     centroids = centroids.reshape(-1, 2)
        #     # flip (x,y) -> (y,x)
        #     centroids = centroids[:, ::-1]
        #     return centroids, img_gray.shape

        # Fallback: detect blobs and refine
        keypoints = self.detector.detect(img_8bit)
        if len(keypoints) == 0:
            # return empty centroids rather than raising - upstream code can handle
            self.logger.warning(
                "No blobs found by detector - returning empty centroids"
            )
            return np.empty((0, 2), dtype=np.float32), img_gray.shape

        centroids = self.subpixel_gaussian_centroid(img_gray, keypoints=keypoints)
        return centroids, img_gray.shape

    @staticmethod
    def subpixel_gaussian_centroid(img, keypoints, win_size=5):
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.array([kp.pt for kp in keypoints])
        xi = np.rint(pts[:, 0]).astype(int)
        yi = np.rint(pts[:, 1]).astype(int)

        img_padded = np.pad(img, pad_width=win_size, mode="constant", constant_values=0)
        xi_pad = xi + win_size
        yi_pad = yi + win_size

        d = np.arange(-win_size, win_size + 1)
        dy, dx = np.meshgrid(d, d, indexing="ij")

        patches = img_padded[
            yi_pad[:, None, None] + dy[None, :, :],
            xi_pad[:, None, None] + dx[None, :, :],
        ].astype(np.float32)

        total = np.sum(patches, axis=(1, 2))
        valid = total > 1e-6

        sum_y = np.sum(patches * dy[None, :, :], axis=(1, 2))
        sum_x = np.sum(patches * dx[None, :, :], axis=(1, 2))

        denom = np.where(valid, total, 1.0)
        cy = yi + sum_y / denom
        cx = xi + sum_x / denom

        refined = np.column_stack((cy[valid], cx[valid]))
        return refined.astype(np.float32)


def _get_default_logger():
    import logging

    logger = logging.getLogger("atommovr.utils.imaging")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

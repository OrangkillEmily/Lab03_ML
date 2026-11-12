"""Visualization helpers for K-Means image segmentation."""

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_rgb_image(image, name="image"):
    """Validate an RGB image."""

    arr = np.asarray(image)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"{name} phải có dạng H x W x 3 (RGB)."
        )

    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(
            f"{name} không được rỗng."
        )

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            f"{name} phải chứa dữ liệu dạng số."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} chứa NaN hoặc Infinity."
        )

    return np.clip(arr, 0, 255).astype(np.uint8)


# ============================================================
# 1. ELBOW PLOT
# ============================================================

def plot_elbow(elbow_df):
    """
    Plot inertia against the number of clusters for each image.

    Parameters
    ----------
    elbow_df : pandas.DataFrame
        Must contain columns:
        - image
        - K
        - inertia

    Returns
    -------
    matplotlib.figure.Figure
    """

    required_columns = {
        "image",
        "K",
        "inertia",
    }

    if elbow_df is None or elbow_df.empty:
        raise ValueError(
            "elbow_df không được rỗng."
        )

    missing_columns = required_columns - set(
        elbow_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"elbow_df thiếu các cột: {sorted(missing_columns)}"
        )

    if elbow_df[["K", "inertia"]].isnull().any().any():
        raise ValueError(
            "elbow_df chứa giá trị K hoặc inertia bị thiếu."
        )

    if (elbow_df["K"] < 1).any():
        raise ValueError(
            "Giá trị K phải lớn hơn hoặc bằng 1."
        )

    if not np.isfinite(
        elbow_df["inertia"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "inertia phải chứa các giá trị hữu hạn."
        )

    images = list(
        elbow_df["image"].dropna().unique()
    )

    if not images:
        raise ValueError(
            "Không tìm thấy tên ảnh hợp lệ trong elbow_df."
        )

    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(6 * len(images), 4),
        squeeze=False,
    )

    for ax, image_name in zip(axes.ravel(), images):

        group = elbow_df[
            elbow_df["image"] == image_name
        ].sort_values("K")

        ax.plot(
            group["K"],
            group["inertia"],
            marker="o",
            linewidth=2,
        )

        ax.set_title(
            f"Elbow Method - {image_name}"
        )

        ax.set_xlabel(
            "Number of clusters (K)"
        )

        ax.set_ylabel(
            "Inertia"
        )

        ax.set_xticks(
            sorted(group["K"].unique())
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    fig.tight_layout()

    return fig


# ============================================================
# 2. SEGMENTATION COMPARISON
# ============================================================

def plot_segmentation_comparison(
    original,
    segmented_by_k,
    image_name="Image",
):
    """
    Compare original image with segmented images.

    Parameters
    ----------
    original : np.ndarray
        Original RGB image.

    segmented_by_k : dict
        Dictionary mapping K to segmented RGB images.

    image_name : str
        Image title.

    Returns
    -------
    matplotlib.figure.Figure
    """

    original = _validate_rgb_image(
        original,
        name="original",
    )

    if not isinstance(segmented_by_k, dict):
        raise TypeError(
            "segmented_by_k phải là dictionary."
        )

    if not segmented_by_k:
        raise ValueError(
            "segmented_by_k không được rỗng."
        )

    sorted_results = sorted(
        segmented_by_k.items(),
        key=lambda item: item[0],
    )

    n_images = len(sorted_results) + 1

    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(5 * n_images, 4),
        squeeze=False,
    )

    axes = axes.ravel()

    # Original image.
    axes[0].imshow(original)

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")

    # Segmented images.
    for ax, (k, segmented) in zip(
        axes[1:],
        sorted_results,
    ):

        segmented = _validate_rgb_image(
            segmented,
            name=f"segmented image K={k}",
        )

        if segmented.shape != original.shape:
            raise ValueError(
                f"K={k}: Kích thước ảnh phân đoạn "
                "không khớp ảnh gốc."
            )

        ax.imshow(segmented)

        ax.set_title(
            f"K-Means (K={k})"
        )

        ax.axis("off")

    fig.suptitle(
        f"Image Segmentation - {image_name}",
        fontsize=13,
    )

    fig.tight_layout()

    return fig


# ============================================================
# 3. COLOR PALETTE
# ============================================================

def plot_color_palette(
    labels,
    centers,
    image_name="Image",
    k=None,
):
    """
    Plot the representative color palette and pixel proportions.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels for all pixels.

    centers : np.ndarray
        RGB cluster centers with shape (K, 3).

    image_name : str
        Image title.

    k : int or None
        Number of clusters. Defaults to len(centers).

    Returns
    -------
    matplotlib.figure.Figure
    """

    labels = np.asarray(labels).ravel()

    centers = np.asarray(
        centers,
        dtype=np.float64,
    )

    if labels.size == 0:
        raise ValueError(
            "labels không được rỗng."
        )

    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(
            "centers phải có dạng K x 3 RGB."
        )

    if len(centers) == 0:
        raise ValueError(
            "centers không được rỗng."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)):
            raise ValueError(
                "labels chứa NaN hoặc Infinity."
            )

        if not np.all(labels == np.floor(labels)):
            raise ValueError(
                "labels phải chứa các số nguyên."
            )

    labels = labels.astype(np.int64)

    if not np.all(np.isfinite(centers)):
        raise ValueError(
            "centers chứa NaN hoặc Infinity."
        )

    if np.any(centers < 0) or np.any(centers > 255):
        raise ValueError(
            "Giá trị RGB của centers phải nằm trong [0, 255]."
        )

    if np.any(labels < 0) or np.any(labels >= len(centers)):
        raise ValueError(
            "labels chứa chỉ số cụm không hợp lệ."
        )

    centers = centers.astype(np.uint8)

    n_clusters = len(centers)

    if k is not None:
        if (
            isinstance(k, bool)
            or not isinstance(k, (int, np.integer))
            or k != n_clusters
        ):
            raise ValueError(
                "k phải bằng số tâm cụm trong centers."
            )

    # Count pixels assigned to each cluster.
    counts = np.bincount(
        labels,
        minlength=n_clusters,
    )

    proportions = counts / counts.sum()

    # Sort palette by pixel proportion, descending.
    order = np.argsort(
        proportions
    )[::-1]

    centers = centers[order]
    counts = counts[order]
    proportions = proportions[order]

    # Plot.
    fig, ax = plt.subplots(
        figsize=(10, 2.5)
    )

    start = 0.0

    for center, proportion, count in zip(
        centers,
        proportions,
        counts,
    ):

        if proportion <= 0:
            continue

        ax.barh(
            y=0,
            width=proportion,
            left=start,
            color=center / 255.0,
            edgecolor="white",
            linewidth=1,
        )

        start += proportion

    ax.set_xlim(0, 1)

    ax.set_yticks([])

    ax.set_xlabel(
        "Proportion of pixels"
    )

    ax.set_title(
        f"Color Palette - {image_name} "
        f"(K={n_clusters})"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    return fig

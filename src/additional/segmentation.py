"""Image segmentation utilities using a from-scratch K-Means algorithm."""

import numpy as np

from .image_io import image_to_pixels, pixels_to_image
from .kmeans import KMeansFromScratch, make_sample


def segment_image(
    image_array,
    n_clusters=5,
    max_fit_pixels=100_000,
    random_state=42,
    max_iter=100,
    tol=1e-3,
    predict_batch_size=50_000,
):
    """
    Segment an RGB image using K-Means.

    The model is fitted on a random sample of pixels to reduce memory
    consumption. After fitting, all pixels are assigned to their nearest
    cluster center.

    Parameters
    ----------
    image_array : numpy.ndarray
        Input image with shape (height, width, 3), RGB format.

    n_clusters : int
        Number of color clusters.

    max_fit_pixels : int
        Maximum number of pixels used to fit K-Means.

    random_state : int
        Random seed for reproducibility.

    max_iter : int
        Maximum number of K-Means iterations.

    tol : float
        Convergence tolerance.

    predict_batch_size : int
        Number of pixels processed in each prediction batch.

    Returns
    -------
    dict
        Dictionary containing:
        - segmented: segmented RGB image.
        - labels: cluster labels with shape (height, width).
        - model: fitted K-Means model.
        - original_shape: original image dimensions.
        - n_pixels: total number of pixels.
        - n_fit_pixels: number of pixels used for fitting.
        - n_clusters: number of clusters.
    """
    # --------------------------------------------------------
    # 1. Validate parameters
    # --------------------------------------------------------

    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
        raise ValueError("n_clusters phải là số nguyên >= 1.")

    if not isinstance(max_fit_pixels, (int, np.integer)) or max_fit_pixels < 1:
        raise ValueError("max_fit_pixels phải là số nguyên >= 1.")

    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter phải là số nguyên >= 1.")

    if not isinstance(predict_batch_size, (int, np.integer)) or predict_batch_size < 1:
        raise ValueError("predict_batch_size phải là số nguyên >= 1.")

    if tol < 0:
        raise ValueError("tol phải >= 0.")

    # --------------------------------------------------------
    # 2. Convert image into RGB pixels
    # --------------------------------------------------------

    pixels, height, width = image_to_pixels(image_array)

    if not np.isfinite(pixels).all():
        raise ValueError("Ảnh đầu vào chứa giá trị NaN hoặc Inf.")

    n_pixels = len(pixels)

    if n_pixels < n_clusters:
        raise ValueError(
            f"Số pixel ({n_pixels}) phải >= số cụm ({n_clusters})."
        )

    # --------------------------------------------------------
    # 3. Sample pixels for model fitting
    # --------------------------------------------------------

    fit_pixels = make_sample(
        pixels,
        max_samples=max_fit_pixels,
        random_state=random_state,
    )

    if len(fit_pixels) < n_clusters:
        raise ValueError(
            "Số pixel dùng để huấn luyện phải >= n_clusters."
        )

    # --------------------------------------------------------
    # 4. Train K-Means
    # --------------------------------------------------------

    model = KMeansFromScratch(
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )

    model.fit(fit_pixels)

    # --------------------------------------------------------
    # 5. Predict cluster labels for all pixels
    # --------------------------------------------------------

    labels = model.predict(
        pixels,
        batch_size=predict_batch_size,
    )

    # --------------------------------------------------------
    # 6. Reconstruct segmented image
    # --------------------------------------------------------

    segmented = pixels_to_image(
        model.cluster_centers_[labels],
        height,
        width,
    )

    labels_image = labels.reshape(height, width)

    # --------------------------------------------------------
    # 7. Return results
    # --------------------------------------------------------

    return {
        "segmented": segmented,
        "labels": labels_image,
        "model": model,
        "original_shape": (height, width, 3),
        "n_pixels": n_pixels,
        "n_fit_pixels": len(fit_pixels),
        "n_clusters": n_clusters,
    }

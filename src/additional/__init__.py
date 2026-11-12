"""Utilities for K-Means image segmentation."""

from .image_io import (
    load_rgb_image,
    image_to_pixels,
    pixels_to_image,
)

from .kmeans import (
    KMeansFromScratch,
    make_sample,
)

from .segmentation import segment_image

from .visualization import (
    plot_elbow,
    plot_segmentation_comparison,
    plot_color_palette,
)

__all__ = [
    "load_rgb_image",
    "image_to_pixels",
    "pixels_to_image",
    "KMeansFromScratch",
    "make_sample",
    "segment_image",
    "plot_elbow",
    "plot_segmentation_comparison",
    "plot_color_palette",
]

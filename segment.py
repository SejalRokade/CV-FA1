from typing import Tuple

import cv2
import numpy as np


def otsu_segment(image_gray: np.ndarray) -> np.ndarray:
    """Otsu thresholding followed by morphological cleanup.

    Returns a binary mask with foreground as 255 and background as 0.
    """
    # Otsu on slightly blurred image for stability
    blur = cv2.GaussianBlur(image_gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure cells are white: invert if needed by comparing mean intensities
    if np.mean(image_gray[mask > 0]) < np.mean(image_gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    # Morphological open to remove speckles, then close to fill holes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep largest connected component to isolate primary cell."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8) * 255



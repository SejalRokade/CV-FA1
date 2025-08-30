from typing import Tuple

import cv2
import numpy as np


def to_grayscale_enhanced(image_bgr: np.ndarray) -> np.ndarray:
    """Convert RGB/uint8 to enhanced grayscale via HSV V-channel + CLAHE.

    We avoid full stain deconvolution since BloodMNIST is normalized; this
    approximates nucleus contrast enhancement while staying lightweight.
    """
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        return v_eq
    elif image_bgr.ndim == 2:
        return image_bgr
    else:
        raise ValueError("Unexpected image shape for grayscale conversion")


def denoise(image_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Light denoising to assist thresholding.

    Bilateral preserves edges better; fallback to Gaussian for speed if needed.
    """
    if ksize <= 1:
        return image_gray
    return cv2.bilateralFilter(image_gray, d=ksize, sigmaColor=50, sigmaSpace=50)



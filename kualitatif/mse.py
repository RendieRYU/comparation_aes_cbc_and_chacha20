import numpy as np


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two images.
    Supports grayscale or color images (uint8 arrays recommended).
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for MSE.")
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    err = np.mean((x - y) ** 2)
    return float(err)

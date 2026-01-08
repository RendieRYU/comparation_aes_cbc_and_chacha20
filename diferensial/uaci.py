import numpy as np


def uaci(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Unified Average Changing Intensity between two images (percentage).
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for UACI.")
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    diff = np.abs(x - y)
    return float(np.mean(diff) / max_val * 100.0)

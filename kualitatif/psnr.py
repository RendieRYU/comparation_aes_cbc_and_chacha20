import math
import numpy as np
from .mse import mse


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.
    """
    m = mse(img1, img2)
    if m == 0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(m)

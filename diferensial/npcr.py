import numpy as np


def npcr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Number of Pixels Change Rate between two images (percentage).
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for NPCR.")
    diff = img1 != img2
    changed = np.count_nonzero(diff)
    total = diff.size
    return float(changed) / float(total) * 100.0

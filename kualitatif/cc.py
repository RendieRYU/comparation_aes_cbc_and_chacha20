import numpy as np


def correlation_coefficient(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Pearson correlation coefficient between two images (flattened).
    Returns value in [-1, 1].
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for correlation.")
    x = img1.astype(np.float64).ravel()
    y = img2.astype(np.float64).ravel()
    if x.size < 2:
        return 0.0
    c = np.corrcoef(x, y)
    return float(c[0, 1])


def adjacent_correlation(image: np.ndarray, direction: str = "horizontal") -> float:
    """
    Correlation between adjacent pixels in a single image.
    direction: 'horizontal', 'vertical', or 'diagonal'
    """
    img = image.astype(np.float64)
    if img.ndim == 3:  # color -> average over channels
        img = img.mean(axis=2)

    if direction == "horizontal":
        a = img[:, :-1].ravel()
        b = img[:, 1:].ravel()
    elif direction == "vertical":
        a = img[:-1, :].ravel()
        b = img[1:, :].ravel()
    elif direction == "diagonal":
        a = img[:-1, :-1].ravel()
        b = img[1:, 1:].ravel()
    else:
        raise ValueError("direction must be 'horizontal', 'vertical', or 'diagonal'")

    if a.size < 2:
        return 0.0
    c = np.corrcoef(a, b)
    return float(c[0, 1])

import numpy as np

try:
    import cv2  # for GaussianBlur
except Exception as e:  # pragma: no cover
    cv2 = None


def _ssim_channel(x: np.ndarray, y: np.ndarray, max_val: float = 255.0) -> float:
    if cv2 is None:
        raise ImportError(
            "OpenCV (opencv-python) is required for SSIM. Please install it."
        )
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # Gaussian window parameters
    K1, K2 = 0.01, 0.03
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    # Use Gaussian blur to compute local means and variances
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12
    )
    return float(ssim_map.mean())


def ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Structural Similarity Index (SSIM) for grayscale or RGB images.
    For color images, returns the average SSIM over channels.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for SSIM.")

    if img1.ndim == 2:
        return _ssim_channel(img1, img2, max_val=max_val)
    elif img1.ndim == 3:
        # Compute per-channel SSIM and average
        scores = []
        for c in range(img1.shape[2]):
            scores.append(_ssim_channel(img1[:, :, c], img2[:, :, c], max_val=max_val))
        return float(np.mean(scores))
    else:
        raise ValueError("Unsupported image dimensions for SSIM.")

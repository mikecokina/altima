import numpy as np


def owa_edge_preserving_smooth_2d(
        data_2d: np.ndarray,
        window_size: int = 9,
        alpha: float = 1.0,
        sigma_d: float = 2.0
) -> np.ndarray:
    # … (unchanged) …
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")
    height, width = data_2d.shape
    half_win = window_size // 2
    padded = np.pad(data_2d, pad_width=half_win, mode='edge')
    coords = np.indices((window_size, window_size))
    center = half_win
    d2 = (coords[0] - center) ** 2 + (coords[1] - center) ** 2
    spatial_kernel = np.exp(-d2 / (2.0 * sigma_d ** 2))
    result = np.zeros_like(data_2d, dtype=np.float32)
    for row in range(height):
        for col in range(width):
            window = padded[row:row + window_size, col:col + window_size]
            local_median = np.median(window)
            abs_devs = np.abs(window - local_median)
            mad = np.median(abs_devs)
            mad = mad if mad >= 1e-12 else 1e-12
            intensity_weight = np.exp(-0.5 * ((window - local_median) / (alpha * mad)) ** 2)
            combined = intensity_weight * spatial_kernel
            combined /= combined.sum()
            result[row, col] = (window * combined).sum()
    return result

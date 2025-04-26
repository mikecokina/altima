#!/usr/bin/env python3
from typing import Tuple, Optional

from PIL import Image
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from skimage.color import rgb2lab


def owa_edge_preserving_smooth_2d(
        data_2d: np.ndarray,
        window_size: int = 9,
        alpha: float = 1.0,
        sigma_d: float = 2.0
) -> np.ndarray:
    """
    Applies an adaptive edge-preserving smoothing filter (OWA-like) to a 2D array.
    - Uses local median + MAD (median absolute deviation) for intensity weighting.
    - Uses a Gaussian spatial weighting with std = sigma_d.
    """
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
            if mad < 1e-12:
                mad = 1e-12
            intensity_weight = np.exp(-0.5 * ((window - local_median) / (alpha * mad)) ** 2)
            combined_weight = intensity_weight * spatial_kernel
            combined_weight /= combined_weight.sum()
            result[row, col] = np.sum(window * combined_weight)
    return result


class TemperatureParserSimple:
    """
    Load a heatmap image (optionally cropped), map every pixel’s RGB color
    to the nearest legend temperature, and return a 2D array of temperatures.
    You can optionally apply either a median or the OWA filter before mapping
    (to remove thin lines) and/or after mapping (to smooth the temperature map).
    """
    rgb_to_temp = {
        (32, 89, 230): -6,
        (16, 107, 242): -5,
        (0, 125, 254): -4,
        (0, 157, 254): -3,
        (0, 190, 255): -2,
        (84, 221, 254): -1,
        (168, 254, 254): 0,
        (84, 250, 225): 1,
        (0, 247, 198): 2,
        (12, 230, 168): 3,
        (24, 214, 139): 4,
        (12, 192, 120): 5,
        (0, 169, 99): 6,
        (22, 169, 72): 7,
        (42, 170, 42): 8,
        (42, 185, 42): 9,
        (43, 200, 43): 10,
        (21, 227, 21): 11,
        (0, 254, 0): 12,
        (101, 254, 0): 13,
        (203, 254, 0): 14,
        (229, 254, 0): 15,
        (254, 254, 0): 16,
        (245, 245, 62): 17,
        (236, 236, 125): 18,
        (232, 220, 114): 19,
        (227, 203, 101): 20,
        (224, 189, 88): 21,
        (219, 173, 72): 22,
        (238, 172, 36): 23,
    }

    @classmethod
    def _build_palette_lab(cls):
        rgb = np.array(list(cls.rgb_to_temp.keys()), dtype=float) / 255.0
        lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        temps = np.array(list(cls.rgb_to_temp.values()), dtype=float)
        tree = cKDTree(lab)
        return tree, temps

    @staticmethod
    def _load_and_crop(img_path: str, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        if bbox:
            img = img.crop(bbox)
        return np.array(img, dtype=np.uint8)

    @classmethod
    def parse_rgb(
            cls,
            img_path: str,
            bbox: Tuple[int, int, int, int] = None,
            *,
            pre_filter: Optional[str] = None,  # 'median', 'owa', or None
            pre_size: int = 3,
            owa_pre_alpha: float = 1.0,
            owa_pre_sigma: float = 2.0,
            post_filter: Optional[str] = None,  # 'median', 'owa', or None
            post_size: int = 3,
            owa_post_alpha: float = 1.0,
            owa_post_sigma: float = 2.0,
    ) -> np.ndarray:
        """
        1) Load & (optional) crop
        2) Optionally apply pre_filter ('median' or 'owa')
        3) Map every pixel’s RGB to the nearest legend temperature in Lab
        4) Optionally apply post_filter ('median' or 'owa')
        Returns a height×width float array of temperatures.
        """
        arr = cls._load_and_crop(img_path, bbox)
        h, w, _ = arr.shape

        # Pre-filter
        if pre_filter == 'median':
            arr = np.stack([
                np.array(median_filter(arr[:, :, 0], size=pre_size), dtype=np.uint8),
                np.array(median_filter(arr[:, :, 1], size=pre_size), dtype=np.uint8),
                np.array(median_filter(arr[:, :, 2], size=pre_size), dtype=np.uint8),
            ], axis=2).astype(np.uint8)
        elif pre_filter == 'owa':
            channels = []
            for c in range(3):
                channels.append(
                    owa_edge_preserving_smooth_2d(
                        arr[:, :, c].astype(float),
                        window_size=pre_size,
                        alpha=owa_pre_alpha,
                        sigma_d=owa_pre_sigma
                    )
                )
            arr = np.stack(channels, axis=2).astype(np.uint8)

        # RGB → Lab → nearest-temperature
        flat_rgb = arr.reshape(-1, 3).astype(float) / 255.0
        lab = rgb2lab(flat_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        tree, temps = cls._build_palette_lab()
        _, idx = tree.query(lab, k=1)
        temp_map = temps[idx].reshape(h, w)

        # Post-filter
        if post_filter == 'median':
            temp_map = median_filter(temp_map, size=post_size)
        elif post_filter == 'owa':
            temp_map = owa_edge_preserving_smooth_2d(
                temp_map,
                window_size=post_size,
                alpha=owa_post_alpha,
                sigma_d=owa_post_sigma
            )

        return temp_map

    @staticmethod
    def rescale_to_255(temp_map: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(temp_map)
        if not np.any(valid):
            return np.zeros_like(temp_map, dtype=np.uint8)
        vmin, vmax = np.nanmin(temp_map), np.nanmax(temp_map)
        span = vmax - vmin if vmax > vmin else 1.0
        scaled = (temp_map - vmin) / span * 255.0
        scaled[~valid] = 0
        return scaled.clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    img_path_ = "/home/mike/Data/meteo/temperature/2025-04-25/T2M_oper_iso_R7_202504251500-0000.png"
    bbox_ = (0, 25, 848, 475)

    temp_map_ = TemperatureParserSimple.parse_rgb(
        img_path_,
        bbox_,
        pre_filter='median',  # 'median', 'owa', or None
        pre_size=5,
        owa_pre_alpha=0.5,
        owa_pre_sigma=1.0,
        post_filter='median',  # 'median', 'owa', or None
        post_size=3
    )

    scaled_ = TemperatureParserSimple.rescale_to_255(temp_map_)
    Image.fromarray(scaled_).convert('RGB').save("temperature_map.png")
    print("Saved temperature_map.png")

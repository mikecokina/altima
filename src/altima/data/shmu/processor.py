#!/usr/bin/env python3
from typing import Tuple

from PIL import Image
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from skimage.color import rgb2lab


class TemperatureParserSimple:
    """
    Load a heatmap image (optionally cropped), map every pixel’s RGB color
    to the nearest legend temperature, and return a 2D array of temperatures.
    Optionally apply a median filter to smooth the result.
    """
    # exact legend RGB → temperature
    rgb_to_temp = {
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
        # …extend if needed…
    }

    @classmethod
    def _build_palette_lab(cls):
        # Build a cKDTree over the Lab values of the legend colors
        rgb = np.array(list(cls.rgb_to_temp.keys()), dtype=float) / 255.0
        lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        temps = np.array(list(cls.rgb_to_temp.values()), dtype=float)
        tree = cKDTree(lab)
        return tree, temps

    @staticmethod
    def _load_and_crop(img_path: str, bbox: Tuple[float, float, float, float] = None) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        if bbox:
            img = img.crop(bbox)
        return np.array(img, dtype=np.uint8)

    @classmethod
    def parse_rgb(
        cls,
        img_path: str,
        bbox: tuple = None,
        apply_median: bool = False,
        median_size: int = 3
    ) -> np.ndarray:
        """
        1) Load & (optional) crop
        2) Map every pixel’s RGB to the nearest legend temperature
        3) Optionally apply a median filter over the temperature map
        Returns a height×width float array of temperatures.
        """
        arr = cls._load_and_crop(img_path, bbox)
        height, width, _ = arr.shape

        # Flatten and normalize
        flat_rgb = arr.reshape(-1, 3).astype(float) / 255.0
        # Convert to Lab
        lab = rgb2lab(flat_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

        # Build (or reuse) the palette tree
        tree, temps = cls._build_palette_lab()
        _, idx = tree.query(lab, k=1)

        # Remap back to 2D
        temp_map = temps[idx].reshape(height, width)

        # Apply median filter if requested
        if apply_median:
            temp_map = median_filter(temp_map, size=median_size)

        return temp_map

    @staticmethod
    def rescale_to_255(temp_map: np.ndarray) -> np.ndarray:
        """
        Linearly stretch non-NaN values of temp_map to [0,255];
        NaNs (if any) become 0.
        """
        valid = ~np.isnan(temp_map)
        if not np.any(valid):
            return np.zeros_like(temp_map, dtype=np.uint8)
        vmin, vmax = np.nanmin(temp_map), np.nanmax(temp_map)
        span = vmax - vmin if vmax > vmin else 1.0
        scaled = (temp_map - vmin) / span * 255.0
        scaled[~valid] = 0
        return scaled.clip(0, 255).astype(np.uint8)


# if __name__ == "__main__":
#     # --- user settings ---
#     img_path_ = "/home/mike/Data/meteo/temperature/2025-04-25/T2M_oper_iso_R7_202504251500-0000.png"
#     bbox_ = (0, 25, 848, 475)
#
#     # parse with optional median filtering
#     temperature_array = TemperatureParserSimple.parse_rgb(
#         img_path_,
#         bbox_,
#         apply_median=True,
#         median_size=3
#     )
#
#     # rescale and save
#     scaled_ = TemperatureParserSimple.rescale_to_255(temperature_array)
#     Image.fromarray(scaled_).convert('RGB').save("temperature_map.png")
#     print("Saved temperature_map.png")

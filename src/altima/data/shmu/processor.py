from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict

from PIL import Image
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from skimage.color import rgb2lab

from altima.data.shmu.utils import owa_edge_preserving_smooth_2d


class BaseParser(ABC):
    """
    Abstract base for any RGB→value heatmap parser.
    Subclasses must define an RGB_TO_VALUE dict.
    """

    RGB_TO_VALUE: Dict[Tuple[int, int, int], float]

    @classmethod
    def _build_palette_lab(cls):
        rgb = np.array(list(cls.RGB_TO_VALUE.keys()), dtype=float) / 255.0
        lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        values = np.array(list(cls.RGB_TO_VALUE.values()), dtype=float)
        tree = cKDTree(lab)
        return tree, values

    @staticmethod
    def _load_and_crop(img_path: str, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        if bbox:
            img = img.crop(bbox)
        return np.array(img, dtype=np.uint8)

    def parse_rgb(
            self,
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
        1) Load & crop image
        2) Optionally pre-filter
        3) Map each pixel to nearest palette value
        4) Optionally post-filter
        """
        arr = self._load_and_crop(img_path, bbox)
        h, w, _ = arr.shape

        # Pre-filter
        if pre_filter == 'median':
            arr = np.stack([
                np.array(median_filter(arr[:, :, c], size=pre_size)).astype(np.uint8)
                for c in range(3)
            ], axis=2)
        elif pre_filter == 'owa':
            arr = np.stack([
                owa_edge_preserving_smooth_2d(arr[:, :, c].astype(float),
                                              window_size=pre_size,
                                              alpha=owa_pre_alpha,
                                              sigma_d=owa_pre_sigma)
                for c in range(3)
            ], axis=2).astype(np.uint8)

        # RGB → Lab → nearest-value
        flat_rgb = (arr.reshape(-1, 3).astype(float) / 255.0)
        lab = rgb2lab(flat_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        tree, values = self._build_palette_lab()
        _, idx = tree.query(lab, k=1)
        val_map = values[idx].reshape(h, w)

        # Post-filter
        if post_filter == 'median':
            val_map = median_filter(val_map, size=post_size)
        elif post_filter == 'owa':
            val_map = owa_edge_preserving_smooth_2d(
                val_map,
                window_size=post_size,
                alpha=owa_post_alpha,
                sigma_d=owa_post_sigma
            )

        return val_map

    @staticmethod
    def rescale_to_255(val_map: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(val_map)
        if not np.any(valid):
            return np.zeros_like(val_map, dtype=np.uint8)
        vmin, vmax = val_map[valid].min(), val_map[valid].max()
        span = vmax - vmin if vmax > vmin else 1.0
        scaled = (val_map - vmin) / span * 255.0
        scaled[~valid] = 0
        return np.clip(scaled, 0, 255).astype(np.uint8)


class TemperatureParser(BaseParser):
    RGB_TO_VALUE = {
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


class HumidityParser(BaseParser):
    RGB_TO_VALUE = {
        (255, 102, 0): 20,
        (254, 126, 0): 25,
        (254, 152, 0): 30,
        (254, 177, 0): 35,
        (254, 203, 0): 40,
        (254, 228, 0): 45,
        (254, 254, 0): 50,
        (209, 240, 1): 55,
        (164, 226, 2): 60,
        (120, 212, 2): 65,
        (75, 198, 3): 70,
        (30, 184, 4): 75,
        (26, 219, 121): 80,
        (23, 255, 240): 85,
        (21, 206, 245): 90,
        (20, 158, 251): 95,
    }


class Rainfall1HParser(BaseParser):
    RGB_TO_VALUE = {
        (255, 255, 255): 0.0,
        (226, 226, 226): 0.1,
        (203, 203, 203): 0.2,
        (189, 189, 189): 0.5,
        (147, 147, 147): 0.8,
        (166, 254, 255): 1.0,
        (0, 255, 255): 1.5,
        (7, 189, 255): 2.0,
        (46, 130, 255): 3.0,
        (0, 85, 255): 4.0,
        (140, 255, 144): 5.0,
        (137, 230, 143): 7.0,
        (86, 214, 125): 10.0,
        (85, 170, 0): 15.0,
        (0, 116, 0): 20.0,
        (214, 255, 33): 25.0,
        (248, 255, 41): 30.0,
        (255, 229, 29): 35.0,
        (255, 170, 127): 40.0,
        (255, 85, 0): 45.0,
        (255, 0, 0): 50.0
    }


class LongLatToXY:
    mapper = {
        # y, x => lat, long
        (120, 812): (49.087964, 22.565644),
        (190, 81): (48.878368, 17.201750),
        (31, 387): (49.613878, 19.467744),
        (406, 312): (47.827059, 18.855583),
        # (119, 473): (49.179446, 20.087768),
    }

    def __init__(self):
        pixel_coords = []
        geo_coords = []

        for (py, px), (lat, lon) in self.mapper.items():
            pixel_coords.append([px, py, 1])  # x, y, 1
            geo_coords.append([lon, lat])  # lon, lat

        pixel_coords = np.array(pixel_coords)
        geo_coords = np.array(geo_coords)

        # Solve least squares for affine transform
        self.affine_matrix, residuals, rank, s = np.linalg.lstsq(pixel_coords, geo_coords, rcond=None)

    def xy_to_latlon(self, x, y):
        px_vec = np.array([x, y, 1])
        lon, lat = px_vec @ self.affine_matrix
        return lat, lon


if __name__ == "__main__":
    img_path_ = "/home/mike/Data/meteo/temperature/2025-04-26/T2M_oper_iso_R7_202504260100-0000.png"
    bbox_ = (0, 25, 848, 475)

    temp_map_ = TemperatureParser().parse_rgb(
        img_path=img_path_,
        bbox=bbox_,
        pre_filter='median',  # 'median', 'owa', or None
        pre_size=5,
        owa_pre_alpha=0.5,
        owa_pre_sigma=1.0,
        post_filter='median',  # 'median', 'owa', or None
        post_size=3
    )

    temp_map_ = TemperatureParser.rescale_to_255(temp_map_)
    Image.fromarray(temp_map_).convert('RGB').save("temperature_map.png")

    p = LongLatToXY()
    print(p.xy_to_latlon(459, 337))

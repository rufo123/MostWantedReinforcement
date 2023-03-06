import cv2
from numpy import ndarray

from strategy.gps.a_gps_strategy import AGPSStrategy


class GPSStrategyCPU(AGPSStrategy):
    def get_gps_mask(self, par_greyscale) -> tuple[ndarray, tuple[int, int]]:
        return self._get_gps_mask_core(par_greyscale)

    def make_gps_contour(self, gps_image, screenshot_to_draw, gps_center) -> list:
        # self.color_hsl_separator(par_screenshot_to_draw)
        tmp_ret, tmp_threshold_cpu = cv2.threshold(gps_image, 102, 255, 0)
        return self._make_gps_contour_core(tmp_threshold_cpu, screenshot_to_draw, gps_center)

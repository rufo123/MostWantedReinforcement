import cv2
from numpy import ndarray

from strategy.gps.a_gps_strategy import AGPSStrategy


class GPSStrategyGPU(AGPSStrategy):
    def get_gps_mask(self, par_greyscale) -> tuple[ndarray, tuple[int, int]]:
        tmp_gps, tmp_gps_center = self._get_gps_mask_core(par_greyscale)
        tmp_gps = cv2.cvtColor(tmp_gps, cv2.COLOR_GRAY2RGB)
        return tmp_gps, tmp_gps_center

    def make_gps_contour(self, gps_image, screenshot_to_draw, gps_center) -> list:
        ret, tmp_threshold_gpu = cv2.cuda.threshold(gps_image, 236, 255, 0)
        tmp_threshold = tmp_threshold_gpu.download()
        return self._make_gps_contour_core(tmp_threshold, screenshot_to_draw, gps_center)

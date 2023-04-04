"""
Module for the implementation of a GPS strategy using GPU acceleration with OpenCV CUDA.
"""

import cv2
from numpy import ndarray

from strategy.gps.a_gps_strategy import AGPSStrategy


class GPSStrategyGPU(AGPSStrategy):
    """
    A class representing a GPS strategy that uses GPU acceleration with OpenCV CUDA.
    """

    def get_gps_mask(self, par_greyscale) -> tuple[ndarray, tuple[int, int], tuple[int, int]]:
        """
        Returns a tuple containing the GPS mask image and its center.

        Parameters:
        - par_greyscale: A greyscale image used to generate the GPS mask.

        Returns:
        - A tuple containing the GPS mask image and its center.
        """
        tmp_gps, tmp_gps_center, tmp_gps_size = self._get_gps_mask_core(par_greyscale)
        tmp_gps = cv2.cvtColor(tmp_gps, cv2.COLOR_GRAY2RGB)
        return tmp_gps, tmp_gps_center, tmp_gps_size

    def make_gps_contour(self, gps_image, screenshot_to_draw, gps_center) -> list:
        """
        Generates the GPS contour from the GPS image.

        Parameters:
        - gps_image: The GPS image used to generate the contour.
        - screenshot_to_draw: The screenshot where the contour should be drawn.
        - gps_center: The center of the GPS image.

        Returns:
        - A list containing the GPS contour.
        """
        _, tmp_threshold_gpu = cv2.cuda.threshold(gps_image, 236, 255, 0)
        tmp_threshold = tmp_threshold_gpu.download()
        return self._make_gps_contour_core(tmp_threshold, screenshot_to_draw, gps_center)

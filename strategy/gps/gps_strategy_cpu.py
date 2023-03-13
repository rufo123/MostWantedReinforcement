"""
This module provides a GPS strategy for CPU-based devices.

This module provides a GPS strategy for CPU-based devices by implementing 
    the AGPSStrategy interface.
"""
import cv2
from numpy import ndarray

from strategy.gps.a_gps_strategy import AGPSStrategy


class GPSStrategyCPU(AGPSStrategy):
    """
    A GPS strategy for CPU-based devices.

    This class provides a GPS strategy for CPU-based devices by implementing the AGPSStrategy 
        interface.

    Methods:
    - get_gps_mask: Returns a tuple containing a NumPy array representing the GPS mask
        and a tuple of integers representing the size of the GPS mask.
    - make_gps_contour: Returns a list of contours representing the GPS mask.
    """

    def get_gps_mask(self, par_greyscale) -> tuple[ndarray, tuple[int, int]]:
        """
        Returns a tuple containing a NumPy array representing the GPS mask and a tuple of 
            integers representing the size of the GPS mask.

        Args:
        - par_greyscale: A NumPy array representing a greyscale image.

        Returns:
        A tuple containing a NumPy array representing the GPS mask and a tuple of integers 
            representing the size of the GPS mask.
        """
        return self._get_gps_mask_core(par_greyscale)

    def make_gps_contour(self, gps_image, screenshot_to_draw, gps_center) -> list:
        """
        Returns a list of contours representing the GPS mask.

        Args:
        - gps_image: A NumPy array representing a GPS image.
        - screenshot_to_draw: A screenshot image.
        - gps_center: A tuple representing the center of the GPS image.

        Returns:
        A list of contours representing the GPS mask.
        """
        # self.color_hsl_separator(par_screenshot_to_draw)
        _, tmp_threshold_cpu = cv2.threshold(gps_image, 102, 255, 0)
        return self._make_gps_contour_core(tmp_threshold_cpu, screenshot_to_draw, gps_center)

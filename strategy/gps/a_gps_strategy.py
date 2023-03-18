"""
This module contains the AGPSStrategy class that is an abstract base class
used for defining GPS-based strategies for video games.

The class contains abstract methods for creating a GPS mask, creating a GPS
contour and for scaling contours.
"""
import math
import sys
from abc import ABC, abstractmethod

import cv2
import numpy as np
from numpy import ndarray


class AGPSStrategy(ABC):
    """
    Abstract Base Class defining GPS-based strategies for video games.
    """

    @abstractmethod
    def get_gps_mask(self, par_greyscale):
        """
        Abstract method that returns a GPS mask given a greyscale image.

        Args:
        - par_greyscale: a greyscale numpy array.

        Returns:
        - a tuple containing the GPS mask and the GPS mask center.
        """

    @abstractmethod
    def make_gps_contour(self, gps_image, screenshot_to_draw, gps_center) -> list:
        """
        Abstract method that returns a GPS contour given a GPS mask and a screenshot.

        Args:
        - gps_image: a numpy array representing the GPS mask.
        - par_screenshot_to_draw: a numpy array representing the game screenshot.
        - par_gps_center: a tuple containing the GPS mask center.

        Returns:
        - a list containing the GPS contour.
        """

    def _make_gps_contour_core(self, par_threshold: ndarray, par_screenshot_to_draw: ndarray,
                               par_gps_center: tuple[int, int]):
        """
        Private method that does the core processing of the make_gps_contour method.

        Args:
        - par_threshold: a numpy array representing the threshold GPS mask.
        - par_screenshot_to_draw: a numpy array representing the game screenshot.
        - par_gps_center: a tuple containing the GPS mask center.

        Returns:
        - a list containing the GPS contour.
        """
        contours, _ = cv2.findContours(par_threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        tmp_min_dist_center_contour = sys.float_info.max
        tmp_closest_contour_index = -1
        center = par_gps_center
        for index, contour in enumerate(contours):
            if (contour is not None) & (cv2.contourArea(contour) > 0):
                tmp_dist_center_contour = abs(
                    cv2.pointPolygonTest(contour, (center[0], center[1]), True))
                if tmp_dist_center_contour <= tmp_min_dist_center_contour:
                    tmp_min_dist_center_contour = tmp_dist_center_contour
                    tmp_closest_contour_index = index

        my_list = list(contours)
        if len(contours) > 0 and tmp_closest_contour_index > -1:
            my_list[tmp_closest_contour_index] = self.__scale_contour(
                contours[tmp_closest_contour_index], 1)
            contours = tuple(my_list)
            cv2.drawContours(par_screenshot_to_draw, contours, tmp_closest_contour_index,
                             (255, 0, 255), -1)
            return contours[tmp_closest_contour_index]
        return None

    def _get_gps_mask_core(self, par_greyscale) -> tuple[ndarray, tuple[int, int]]:
        """
        Private method - Compute the GPS mask for the given grayscale image.

        Args:
            par_greyscale: A numpy array representing a grayscale image.

        Returns:
            A tuple containing the GPS mask and its center point.
        """

        # gps = np.zeros_like(par_greyscale)
        # par_gps_center = (
        #    par_greyscale.shape[1] - math.ceil(par_greyscale.shape[1] / 1.182),
        #    par_greyscale.shape[0] - math.ceil(par_greyscale.shape[0] / 4.8))
        # gps = cv2.ellipse(gps, par_gps_center,
        #                  (math.ceil(par_greyscale.shape[1] / 11), math.ceil(
        #                  par_greyscale.shape[0] / 8.4)), angle=0,
        #                  startAngle=180, endAngle=360, color=255, thickness=-1)
        # return gps, par_gps_center

        # Compute the center of the GPS mask
        gps_center = (
            par_greyscale.shape[1] - math.ceil(par_greyscale.shape[1] / 1.182),
            par_greyscale.shape[0] - math.ceil(par_greyscale.shape[0] / 4.8))

        # Compute the size of the GPS mask
        gps_size = (math.ceil(par_greyscale.shape[1] / 11), math.ceil(par_greyscale.shape[0] / 8.4))

        # Draw the ellipse on the GPS mask using numpy
        gps = np.zeros_like(par_greyscale)
        gps = cv2.ellipse(gps, gps_center, gps_size, 0, 180, 360, color=255, thickness=-1)
        return gps, gps_center

    def __scale_contour(self, par_cnt, par_scale):
        """
        Protected Method - Scale a contour by a given factor around its center point.

        Args:
            par_cnt: A numpy array representing a contour.
            par_scale: A float representing the scaling factor.

        Returns:
            A numpy array representing the scaled contour.
        """
        if par_cnt is not None:
            moments = cv2.moments(par_cnt)
            m00 = moments['m00']
            if m00 == 0:
                return None
            c_x = int(moments['m10'] / m00)
            c_y = int(moments['m01'] / m00)

            cnt_norm = par_cnt - [c_x, c_y]
            cnt_scaled = cnt_norm * par_scale + [c_x, c_y]
            return cnt_scaled.astype(np.int32)
        return None

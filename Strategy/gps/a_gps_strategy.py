import math
import sys
from abc import ABC, abstractmethod
import cv2
import numpy as np
from numpy import ndarray


class AGPSStrategy(ABC):
    @abstractmethod
    def get_gps_mask(self, par_greyscale):
        pass

    @abstractmethod
    def make_gps_contour(self, gps_image, screenshot_to_draw, gps_center) -> list:
        pass

    def _make_gps_contour_core(self, par_threshold, screenshot_to_draw, gps_center):
        contours, hierarchy = cv2.findContours(par_threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        tmp_min_dist_center_contour = sys.float_info.max
        tmp_closest_contour_index = -1
        center = gps_center
        for index, contour in enumerate(contours):
            if (contour is not None) & (cv2.contourArea(contour) > 0):
                tmp_dist_center_contour = abs(cv2.pointPolygonTest(contour, (center[0], center[1]), True))
                if tmp_dist_center_contour <= tmp_min_dist_center_contour:
                    tmp_min_dist_center_contour = tmp_dist_center_contour
                    tmp_closest_contour_index = index

        my_list = list(contours)
        if len(contours) > 0 and tmp_closest_contour_index > -1:
            my_list[tmp_closest_contour_index] = self.__scale_contour(contours[tmp_closest_contour_index], 1)
            contours = tuple(my_list)
            cv2.drawContours(screenshot_to_draw, contours, tmp_closest_contour_index, (255, 0, 255), -1)
            return contours[tmp_closest_contour_index]
        else:
            return None

    def _get_gps_mask_core(self, par_greyscale) -> tuple[ndarray, tuple[int, int]]:
        # gps = np.zeros_like(par_greyscale)
        # gps_center = (
        #    par_greyscale.shape[1] - math.ceil(par_greyscale.shape[1] / 1.182),
        #    par_greyscale.shape[0] - math.ceil(par_greyscale.shape[0] / 4.8))
        # gps = cv2.ellipse(gps, gps_center,
        #                  (math.ceil(par_greyscale.shape[1] / 11), math.ceil(par_greyscale.shape[0] / 8.4)), angle=0,
        #                  startAngle=180, endAngle=360, color=255, thickness=-1)
        # return gps, gps_center

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

    def __scale_contour(self, cnt, scale):
        if cnt is not None:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                return None
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cnt_norm = cnt - [cx, cy]
            cnt_scaled = cnt_norm * scale
            cnt_scaled = cnt_scaled + [cx, cy]
            cnt_scaled = cnt_scaled.astype(np.int32)
            return cnt_scaled
        else:
            return None

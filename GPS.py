import math
import sys

import cv2
import numpy as np
from numpy import ndarray

from strategy.gps.a_gps_strategy import AGPSStrategy
from strategy.gps.gps_strategy_cpu import GPSStrategyCPU
from strategy.gps.gps_strategy_enum import GPSStrategyEnum
from strategy.gps.gps_strategy_gpu import GPSStrategyGPU


class GPS:
    a_last_calculated_distance_offset: float
    a_last_calculated_direction_offset: int

    a_gps_strategy: AGPSStrategy

    def __init__(self, par_strategy_enum: GPSStrategyEnum):
        self.a_last_calculated_distance_offset = 0
        self.a_last_calculated_direction_offset = 0

        if par_strategy_enum == GPSStrategyEnum.CPU:
            self.a_gps_strategy = GPSStrategyCPU()
        else:
            self.a_gps_strategy = GPSStrategyGPU()

    def make_gps_contour(self, par_gps_image: ndarray, par_screenshot: ndarray, par_gps_center) -> list:
        return self.a_gps_strategy.make_gps_contour(par_gps_image, par_screenshot, par_gps_center)

    def get_gps_mask(self, par_greyscale) -> tuple[ndarray, tuple[int, int]]:
        return self.a_gps_strategy.get_gps_mask(par_greyscale)

    def get_car_point(self, screenshot_to_draw, gps_center):
        tmp_offset = screenshot_to_draw.shape[1] * 0.012
        cv2.circle(screenshot_to_draw, (int(gps_center[0]), int(gps_center[1] - tmp_offset)), 1, (0, 0, 0), 1)
        return np.array([float(gps_center[0]), float(gps_center[1] - tmp_offset)])

    def generate_rgb_255(self, index, pos):
        test = 50 + (pos * 5) * index
        return test

    def polygon_contour_test(self, par_contour, par_car_pos):
        if par_contour is not None and par_car_pos is not None and len(par_car_pos) > 1:
            self.a_last_calculated_distance_offset = cv2.pointPolygonTest(par_contour,
                                                                          (float(par_car_pos[0]), float(par_car_pos[1])), True)
            #print(self.a_last_calculated_distance_offset)
        return self.a_last_calculated_distance_offset

    def check_direction_point_to_contour(self, par_contour, par_car_pos) -> int:
        if par_contour is not None and par_car_pos is not None and len(par_car_pos) > 1:
            nearest_pt_idx = np.argmin(np.linalg.norm(par_contour - par_car_pos, axis=2))
            if par_contour[nearest_pt_idx][0][0] < par_car_pos[0]:
                #print(1)
                # Car is inclined to the right of the road centre
                self.a_last_calculated_direction_offset = 1
            elif par_contour[nearest_pt_idx][0][0] > par_car_pos[0]:
                #print(0)
                # Car is inclined to the left of the road centre
                self.a_last_calculated_direction_offset = -1
            else:
                # Car is going straight towards the road centre
                self.a_last_calculated_direction_offset = 0
        return self.a_last_calculated_direction_offset

    def translate_direction_offset_to_string(self, par_direction_from_contour: int) -> str:
        if par_direction_from_contour == -1:
            return "Left"
        elif par_direction_from_contour == 1:
            return "Right"
        else:
            return "Straight"

    def divide_if_possible(self, dividend, divider):
        if (divider is not None) & (divider != 0):
            return dividend / divider

    def nothing(self, x):
        pass

    def color_grey_separator(self, image):
        # Create a window
        cv2.namedWindow('image')

        # Create trackbars for color change
        cv2.createTrackbar('Min', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('Max', 'image', 0, 255, self.nothing)

        gray_min = gray_max = 0

        while (1):
            # Get current positions of all trackbars
            gray_min = cv2.getTrackbarPos('Min', 'image')
            gray_max = cv2.getTrackbarPos('Max', 'image')

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.equalizeHist(hsv)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(hsv)

            # cv2.imshow('jeje', equalized)
            mask = cv2.inRange(equalized, gray_min, gray_max)
            result = cv2.bitwise_and(image, image, mask=mask)

            # Display result image
            cv2.imshow('image', result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def color_hsl_separator(self, image):
        # Create a window
        cv2.namedWindow('image')

        # Create trackbars for color change
        # Hue is from 0-179 for Opencv
        cv2.createTrackbar('HMin', 'image', 0, 179, self.nothing)
        cv2.createTrackbar('SMin', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('VMin', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('HMax', 'image', 0, 179, self.nothing)
        cv2.createTrackbar('SMax', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('VMax', 'image', 0, 255, self.nothing)

        # Set default value for Max HSV trackbars
        cv2.setTrackbarPos('HMax', 'image', 179)
        cv2.setTrackbarPos('SMax', 'image', 255)
        cv2.setTrackbarPos('VMax', 'image', 255)

        # Initialize HSV min/max values
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0

        while (1):
            # Get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin', 'image')
            sMin = cv2.getTrackbarPos('SMin', 'image')
            vMin = cv2.getTrackbarPos('VMin', 'image')
            hMax = cv2.getTrackbarPos('HMax', 'image')
            sMax = cv2.getTrackbarPos('SMax', 'image')
            vMax = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and maximum HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)

            # Print if there is a change in HSV value
            if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (
                    pvMax != vMax)):
                print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
                    hMin, sMin, vMin, hMax, sMax, vMax))
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax

            # Display result image
            cv2.imshow('image', result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

import math
import sys

import cv2
import numpy as np


class GPS:

    def __init__(self):
        pass

    def get_gps_mask(self, par_greyscale):
        gps = np.zeros_like(par_greyscale)
        gps_center = (
            par_greyscale.shape[1] - math.ceil(par_greyscale.shape[1] / 1.182),
            par_greyscale.shape[0] - math.ceil(par_greyscale.shape[0] / 4.8))
        gps = cv2.ellipse(gps, gps_center,
                          (math.ceil(par_greyscale.shape[1] / 11), math.ceil(par_greyscale.shape[0] / 8.4)), angle=0,
                          startAngle=180, endAngle=360, color=255, thickness=-1)
        return gps, gps_center

    def make_gps_contour(self, gps_image, screenshot_to_draw, gps_center):
        # self.color_hsl_separator(screenshot_to_draw)
        # ret, tresh = cv2.threshold(gps_image, 102, 255, 0)
        ret, tresh = cv2.threshold(gps_image, 236, 255, 0)
        contours, hierarchy = cv2.findContours(tresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

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
        my_list[tmp_closest_contour_index] = self.scale_contour(contours[tmp_closest_contour_index], 1)
        contours = tuple(my_list)

        cv2.drawContours(screenshot_to_draw, contours, tmp_closest_contour_index, (255, 0, 255), -1)


        return contours[tmp_closest_contour_index]

    def get_car_point(self, screenshot_to_draw, gps_center):
        tmp_offset = screenshot_to_draw.shape[1] * 0.012
        cv2.circle(screenshot_to_draw, (int(gps_center[0]), int(gps_center[1] - tmp_offset)), 1, (0, 0, 0), 1)
        return np.array([float(gps_center[0]), float(gps_center[1] - tmp_offset)])

    def scale_contour(self, cnt, scale):
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)

        return cnt_scaled

    def generate_rgb_255(self, index, pos):
        test = 50 + (pos * 5) * index
        return test

    def polygon_contour_test(self, par_contour, par_car_pos):
        return cv2.pointPolygonTest(par_contour, (float(par_car_pos[0]), float(par_car_pos[1])), True)

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

import cv2
import numpy as np
from numpy import ndarray

from GPS import GPS
from Strategy.gps_image_recognition.a_gps_ircgn_strategy import AGpsImageRecognitionStrategy


class GpsImageRecognitionStrategyCPU(AGpsImageRecognitionStrategy):

    import time

    def calc_car_offset(self, par_gps: GPS, par_image: ndarray) -> tuple[float, list]:
        time = self.time.time()
        # Convert to grayscale and equalize histogram
        tmp_grayscale_gpu = self.make_grayscale(par_image)
        # tmp_gps_final = cv2.equalizeHist(tmp_grayscale)

        # tmp_equalized_gpu = cv2.cuda.equalizeHist(tmp_grayscale_gpu)

        # tmp_gps_final = tmp_equalized_gpu.download()

        # Get GPS mask
        tmp_gps, tmp_gps_center = par_gps.get_gps_mask(tmp_grayscale_gpu)

        # Apply GPS mask to screenshot

        tmp_gps = cv2.bitwise_and(par_image, par_image, mask=tmp_gps)

        print(str(self.time.time() - time))

        cv2.imshow('Main DKO', tmp_gps)

        # Convert to HSV and create grayscale mask
        tmp_gps_hsv = cv2.cvtColor(tmp_gps, cv2.COLOR_BGR2HSV)

        tmp_lower_bound = np.array([0, 0, 174])
        tmp_upper_bound = np.array([197, 10, 255])

        tmp_gps_greyscale = cv2.inRange(tmp_gps_hsv, tmp_lower_bound, tmp_upper_bound)


        # Create GPS contour
        tmp_contour = par_gps.make_gps_contour(tmp_gps_greyscale, par_image, tmp_gps_center)

        # car position
        tmp_car_pos = par_gps.get_car_point(par_image, tmp_gps_center)

        # Get car offset and lap progress
        tmp_car_offset = par_gps.polygon_contour_test(tmp_contour, tmp_car_pos)

        return tmp_car_offset, tmp_contour

    def make_grayscale(self, par_image: ndarray) -> cv2.cuda.GpuMat:
        grayscale = cv2.cvtColor(par_image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(grayscale, (5, 5), 0)
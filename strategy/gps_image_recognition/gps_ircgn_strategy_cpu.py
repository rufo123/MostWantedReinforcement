"""
This module provides a CPU implementation of a GPS image recognition strategy.

It includes one class:
- GpsImageRecognitionStrategyCPU: A class that inherits from AGpsImageRecognitionStrategy and
    implements its methods.

This module depends on the following packages:
- cv2: An open-source computer vision library.
- numpy: A package for scientific computing with Python.
"""
import cv2
import numpy as np
from numpy import ndarray

from gps import GPS
from strategy.gps_image_recognition.a_gps_ircgn_strategy import AGpsImageRecognitionStrategy


class GpsImageRecognitionStrategyCPU(AGpsImageRecognitionStrategy):
    """
    A class that implements the AGpsImageRecognitionStrategy using a CPU implementation.
    """

    def __init__(self):
        """
        Constructor for GpsImageRecognitionStrategyCPU class.

        Instance Attributes:
            - a_car_offset (None): The offset of the car, relative to the center of the GPS contour.
        """
        self.a_car_offset = None

    def calc_car_offset(self, par_gps: GPS, par_image: ndarray) -> tuple[float, list, int]:
        """
        Calculates the offset of the car relative to the center of the GPS contour and the progress
            of the current lap.

        Args:
            - par_gps (GPS): The GPS instance used to calculate the car offset.
            - par_image (ndarray): The image containing the car.

        Returns:
            A tuple containing the following:
            - tmp_car_offset_dist (float): The offset of the car, relative to the center of the GPS
                contour.
            - tmp_contour (list): The GPS contour.
            - tmp_car_directional_offset (int): The directional offset of the car, relative to the 
                GPS contour.

        """
        # Convert to grayscale and equalize histogram
        tmp_grayscale_gpu = self.make_grayscale(par_image)
        # tmp_gps_final = cv2.equalizeHist(tmp_grayscale)

        # tmp_equalized_gpu = cv2.cuda.equalizeHist(tmp_grayscale_gpu)

        # tmp_gps_final = tmp_equalized_gpu.download()

        # Get GPS mask
        tmp_gps, tmp_gps_center = par_gps.get_gps_mask(tmp_grayscale_gpu)

        # Apply GPS mask to screenshot

        tmp_gps = cv2.bitwise_and(par_image, par_image, mask=tmp_gps)

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
        tmp_car_offset_dist = par_gps.polygon_contour_test(tmp_contour, tmp_car_pos)

        tmp_car_directional_offset: int = par_gps.check_direction_point_to_contour(tmp_contour,
                                                                                   tmp_car_pos)

        self.a_car_offset = tmp_car_offset_dist

        return tmp_car_offset_dist, tmp_contour, tmp_car_directional_offset

    def make_grayscale(self, par_image: ndarray) -> cv2.cuda.GpuMat:
        """
        Convert an image to grayscale and apply a Gaussian blur.

        Args:
            par_image (ndarray): A NumPy array representing an image.

        Returns:
            cv2.cuda.GpuMat: A CUDA-based representation of the grayscale image.
        """
        grayscale = cv2.cvtColor(par_image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(grayscale, (5, 5), 0)

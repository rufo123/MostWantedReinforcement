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

    def gps_data_with_greyscale(self, par_gps: GPS, par_image: ndarray) -> \
            tuple[ndarray, ndarray, tuple[int, int], tuple[int, int]]:
        """
        Converts a color image to grayscale, applies a GPS mask to it, and returns the resulting
         grayscale image and GPS mask.

        Args:
            par_gps: An instance of the GPS class representing the GPS data to apply to the
             grayscale image.
            par_image: A numpy array representing the color image to convert to grayscale.

        Returns:
            A tuple containing:
            - The resulting grayscale image as a numpy array.
            - The GPS mask as a numpy array.
            - A tuple containing the (x,y) coordinates of the center of the GPS mask in the
             grayscale image.
        """
        # Convert to grayscale and equalize histogram
        tmp_grayscale_gpu = self.make_grayscale(par_image)
        # tmp_gps_final = cv2.equalizeHist(tmp_grayscale)

        # tmp_equalized_gpu = cv2.cuda.equalizeHist(tmp_grayscale_gpu)

        # tmp_gps_final = tmp_equalized_gpu.download()

        # Get Grayscale GPS Mask

        # Get GPS mask
        tmp_gps, tmp_gps_center, tmp_gps_size = par_gps.get_gps_mask(tmp_grayscale_gpu)

        return tmp_grayscale_gpu, tmp_gps, tmp_gps_center, tmp_gps_size

    # pylint:disable= too-many-arguments
    def get_half_gps_greyscale(self, par_image: ndarray,
                               par_grayscale_image: ndarray,
                               par_gps_mask: ndarray,
                               par_gps_center: tuple[int, int],
                               par_gps_size: tuple[int, int]) \
            -> ndarray:
        """
        Applies a GPS mask to a grayscale image and returns the resulting image cropped to the GPS
         mask boundaries.

        Args:
            par_grayscale_image: A numpy array representing the grayscale image to apply the GPS
             mask to.
            par_gps_mask: A numpy array representing the GPS mask to apply to the grayscale image.
            par_image: A numpy array containing image from a game.
            par_gps_center: 
            par_gps_size:

        Returns:
            A numpy array representing the resulting cropped grayscale image after applying the
             GPS mask.
        """
        # Get Gps Mask in Greyscale

        # 1. Apply Mask to Greyscale
        tmp_gps_greyscale = \
            cv2.bitwise_and(par_grayscale_image, par_grayscale_image, mask=par_gps_mask)

        # 2. Get bounding box around gps greyscale where are non 0 pixels
        coord_x = int(par_gps_center[0] - par_gps_size[0])
        coord_y = int(par_gps_center[1] - par_gps_size[1])
        coord_width = int(par_gps_size[0] * 2)
        coord_height = int(par_gps_size[1])

        cropped_gps_greyscale = \
            tmp_gps_greyscale[coord_y:coord_y + coord_height, coord_x:coord_x + coord_width]

        cv2.rectangle(par_image, (coord_x, coord_y),
                      (coord_x + coord_width, coord_y + coord_height),
                      (255, 0, 255), 2)

        cv2.resize(cropped_gps_greyscale, (48, 48))

        return cropped_gps_greyscale

    def calc_car_offset(self, par_gps: GPS, par_image: ndarray,
                        par_gps_mask: ndarray, par_gps_center: tuple[int, int]
                        ) \
            -> tuple[float, list, int]:
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

        # Apply GPS mask to screenshot

        par_gps_mask = cv2.bitwise_and(par_image, par_image, mask=par_gps_mask)

        # Convert to HSV and create grayscale mask
        tmp_gps_hsv = cv2.cvtColor(par_gps_mask, cv2.COLOR_BGR2HSV)

        tmp_lower_bound = np.array([0, 0, 174])
        tmp_upper_bound = np.array([197, 10, 255])

        tmp_gps_greyscale = cv2.inRange(tmp_gps_hsv, tmp_lower_bound, tmp_upper_bound)

        # Create GPS contour
        tmp_contour = par_gps_mask.make_gps_contour(tmp_gps_greyscale, par_image, par_gps_center)

        # car position
        tmp_car_pos = par_gps_mask.get_car_point(par_image, par_gps_center)

        # Get car offset and lap progress
        tmp_car_offset_dist = par_gps_mask.polygon_contour_test(tmp_contour, tmp_car_pos)

        tmp_car_directional_offset: int = par_gps_mask.check_direction_point_to_contour(tmp_contour,
                                                                                        tmp_car_pos)

        self.a_car_offset = tmp_car_offset_dist

        return tmp_car_offset_dist, tmp_contour, tmp_car_directional_offset

    def make_grayscale(self, par_image: ndarray) -> np.ndarray:
        """
        Convert an image to grayscale and apply a Gaussian blur.

        Args:
            par_image (ndarray): A NumPy array representing an image.

        Returns:
            cv2.cuda.GpuMat: A CUDA-based representation of the grayscale image.
        """
        grayscale = cv2.cvtColor(par_image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(grayscale, (5, 5), 0)

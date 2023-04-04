""" 
Module providing Abstract Base class for implementing a strategy to recognize the position
    of a car in a racing game based on GPS data and a screenshot of the game
"""
from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy
from numpy import ndarray

from gps import GPS


class AGpsImageRecognitionStrategy(ABC):
    """
    Abstract base class for implementing a strategy to recognize the position
    of a car in a racing game based on GPS data and a screenshot of the game.
    """

    @abstractmethod
    def gps_data_with_greyscale(self, par_gps: GPS, par_image: ndarray) -> \
            tuple[ndarray, ndarray, tuple[int, int], tuple[int, int]]:
        """
        Converts a color image to grayscale, applies a GPS mask to it, and returns the resulting
         grayscale image and GPS mask.

        
         :param par_gps: An instance of the GPS class representing the GPS data to apply to the
            grayscale image.
         :param par_image: A numpy array representing the color image to convert to grayscale.

        
        :returns: A tuple containing:
            - The resulting grayscale image as a numpy array.
            - The GPS mask as a numpy array.
            - A tuple containing the (x,y) coordinates of the center of the GPS mask in the
             grayscale image.
        """

    # pylint:disable=too-many-arguments
    @abstractmethod
    def get_half_gps_greyscale(self, par_image: ndarray,
                               par_grayscale_image: ndarray,
                               par_gps_mask: ndarray,
                               par_gps_center: tuple[int, int],
                               par_gps_size: tuple[int, int]) \
            -> ndarray:
        """
        Applies a GPS mask to a grayscale image and returns the resulting image cropped to the GPS
         mask boundaries.

         :param par_gps_size:
         :param par_gps_center: 
         :param par_image: A numpy array containing image from a game.
         :param par_grayscale_image: A numpy array representing the grayscale image to apply the GPS
             mask to.
         :param par_gps_mask: A numpy array representing the GPS mask to apply to the grayscale
            image.

        :returns: A numpy array representing the resulting cropped grayscale image after applying
            theGPS mask.
            
        """

    @abstractmethod
    def calc_car_offset(self, par_gps: GPS, par_image: ndarray,
                        par_gps_mask: ndarray,
                        par_gps_center: tuple[int, int]) -> \
            tuple[float, list, int]:
        """
        Abstract Method That Calculates Car Offset From Minimap (GPS)
        :param par_gps_center: Tuple containing x and y coordinates of gps center.
        :param par_gps_mask: Image representing gps/mask
        :param par_gps: The GPS Class
        :param par_image: Screenshot of Game Screen as ndarray
        :returns: A tuple containing the following:
            - float: The offset of the car, relative to the center
                    of the GPS contour.
            - list: The GPS contour.
            - int: The directional offset of the car, 
                        relative to the GPS contour.
        """

    # noinspection PyUnresolvedReferences
    @abstractmethod
    def make_grayscale(self, par_image: ndarray) -> Union[cv2.cuda_GpuMat, numpy.ndarray]:
        """
        Creates GrayScale GpuMat Object from the Image as ndarray
        and then does also GaussianBlur
        :param par_image: Screenshot of Game Screen as ndarray
        :return: either a cv2.cuda_GpuMat or ndarray object
        """

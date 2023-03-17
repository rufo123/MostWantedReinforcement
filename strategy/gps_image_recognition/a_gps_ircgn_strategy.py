""" Module providing Abstract Base Classes"""
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
    def calc_car_offset(self, par_gps: GPS, par_image: ndarray) -> tuple[float, list, int]:
        """
        Abstract Method That Calculates Car Offset From Minimap (GPS)
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

from abc import ABC, abstractmethod
import cv2
from numpy import ndarray

from GPS import GPS


class AGpsImageRecognitionStrategy(ABC):
    @abstractmethod
    def calc_car_offset(self, par_gps: GPS, par_image: ndarray) -> tuple[float, list, int]:
        """
        Abstract Method That Calculates Car Offset From Minimap (GPS)
        :param par_gps: The GPS Class
        :param par_image: Screenshot of Game Screen as ndarray
        """
        pass

    @abstractmethod
    def make_grayscale(self, par_image: ndarray) -> cv2.cuda.GpuMat:
        """
        Creates GrayScale GpuMat Object from the Image as ndarray
        and then does also GaussianBlur
        :param par_image: Screenshot of Game Screen as ndarray
        """
        pass

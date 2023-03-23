"""
This module provides a class representing a GPS object with the ability to calculate
    distance and direction offsets using CPU or GPU strategy.
"""
import cv2
import numpy as np
from numpy import ndarray

from strategy.gps.a_gps_strategy import AGPSStrategy
from strategy.gps.gps_strategy_cpu import GPSStrategyCPU
from strategy.gps.gps_strategy_enum import GPSStrategyEnum
from strategy.gps.gps_strategy_gpu import GPSStrategyGPU


class GPS:
    """
    A class representing a GPS object with the ability to calculate distance and direction offsets 
        using CPU or GPU strategy.

    Attributes:
    a_last_calculated_distance_offset (float): The last calculated distance offset.
    a_last_calculated_direction_offset (int): The last calculated direction offset.
    a_gps_strategy (AGPSStrategy): The GPS strategy object to use for calculating distance and 
        direction offsets.

    Methods:
    init(self, par_strategy_enum: GPSStrategyEnum): Constructs a GPS object with the specified
        strategy type.

    Parameters:
    par_strategy_enum (GPSStrategyEnum): The GPS strategy type to use.

    Returns:
    GPS: A GPS object with the specified strategy type.
    """
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

    def make_gps_contour(self, par_gps_image: ndarray, par_screenshot: ndarray,
                         par_gps_center) -> list:
        """
        Generates a GPS contour based on the GPS image and screenshot.

        Args:
            par_gps_image (ndarray): A GPS image as a ndarray.
            par_screenshot (ndarray): A screenshot as a ndarray.
            par_gps_center: The center of the GPS in the screenshot.

        Returns:
            list: A list of contours that represent the GPS.
        """
        return self.a_gps_strategy.make_gps_contour(par_gps_image, par_screenshot, par_gps_center)

    def get_gps_mask(self, par_greyscale) -> tuple[ndarray, tuple[int, int]]:
        """
        Generates a GPS mask based on a greyscale image.

        Args:
            par_greyscale: A greyscale image.

        Returns:
            tuple[ndarray, tuple[int, int]]: A tuple containing the GPS mask and its dimensions.
        """
        return self.a_gps_strategy.get_gps_mask(par_greyscale)

    def get_car_point(self, par_screenshot_to_draw: ndarray,
                      par_gps_center: tuple[int, int]) -> ndarray:
        """
        Gets the location of the car on the screenshot.

        Args:
            par_screenshot_to_draw: The screenshot to draw on.
            par_gps_center: The center of the GPS.

        Returns:
            ndarray: An array representing the location of the car.
        """
        tmp_offset = par_screenshot_to_draw.shape[1] * 0.012
        cv2.circle(par_screenshot_to_draw, (int(par_gps_center[0]), int(par_gps_center[1] -
                                                                        tmp_offset)), 1,
                   (0, 0, 0), 1)
        return np.array([float(par_gps_center[0]), float(par_gps_center[1] - tmp_offset)])

    def polygon_contour_test(self, par_contour: list, par_car_pos: ndarray) -> float:
        """
        Calculates the distance offset of the car from the GPS contour.

        Args:
            par_contour: The GPS contour.
            par_car_pos: The position of the car.

        Returns:
            float: The distance offset of the car from the GPS contour.
        """
        if par_contour is not None and par_car_pos is not None and len(par_car_pos) > 1:
            self.a_last_calculated_distance_offset = cv2.pointPolygonTest(par_contour,
                                                                          (
                                                                              float(par_car_pos[0]),
                                                                              float(
                                                                                  par_car_pos[1])),
                                                                          True)
            # print(self.a_last_calculated_distance_offset)
        return self.a_last_calculated_distance_offset

    def check_direction_point_to_contour(self, par_contour: list, par_car_pos: ndarray) -> int:
        """
        Checks the direction of the car relative to the GPS contour.

        Args:
            par_contour: The GPS contour.
            par_car_pos: The position of the car.

        Returns:
            int: An integer representing the direction of the car relative to the GPS contour.
                 0: Straight
                 1: Right
                -1: Left
        """
        if par_contour is not None and par_car_pos is not None and len(par_car_pos) > 1:
            nearest_pt_idx = np.argmin(np.linalg.norm(par_contour - par_car_pos, axis=2))
            if par_contour[nearest_pt_idx][0][0] < par_car_pos[0]:
                # print(1)
                # Car is inclined to the right of the road centre
                self.a_last_calculated_direction_offset = 1
            elif par_contour[nearest_pt_idx][0][0] > par_car_pos[0]:
                # print(0)
                # Car is inclined to the left of the road centre
                self.a_last_calculated_direction_offset = -1
            else:
                # Car is going straight towards the road centre
                self.a_last_calculated_direction_offset = 0
        return self.a_last_calculated_direction_offset

    def translate_direction_offset_to_string(self, par_direction_from_contour: int) -> str:
        """
        Translates the direction offset to a string representation.

        Parameters:
        par_direction_from_contour (int): The direction offset value. Must be -1, 1, or 0.

        Returns:
        str: The string representation of the direction offset.
        """
        if par_direction_from_contour == -1:
            return "Left"
        if par_direction_from_contour == 1:
            return "Right"
        return "Straight"

    def divide_if_possible(self, par_dividend: float, par_divider: float) -> float:
        """
        Divides the par_dividend by the par_divider if the par_divider is not None and not zero.

        Parameters:
        par_dividend (float): The par_dividend value.
        par_divider (float): The par_divider value.

        Returns:
        float: The quotient of the division if the par_divider is not None and not zero, otherwise 
            None.
        """
        if (par_divider is not None) & (par_divider != 0):
            return par_dividend / par_divider
        raise ValueError("Cannot divide by zero / divider is unknown")

    def nothing(self, par_x):
        """
        A placeholder function that does nothing.

        Parameters:
        x: Any value. This parameter is ignored.

        Returns:
        None
        """

    def color_grey_separator(self, par_image: ndarray) -> None:
        """
        Creates a window with trackbars for color change, and allows the user to separate a 
            grayscale image into two different colors
        based on the threshold values set by the trackbars.

        Parameters:
        par_image (numpy.ndarray): The grayscale image to be separated.

        Returns:
        None
        """
        # Create a window
        cv2.namedWindow('image')

        # Create trackbars for color change
        cv2.createTrackbar('Min', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('Max', 'image', 0, 255, self.nothing)

        while 1:
            # Get current positions of all trackbars
            gray_min = cv2.getTrackbarPos('Min', 'image')
            gray_max = cv2.getTrackbarPos('Max', 'image')

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(par_image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.equalizeHist(hsv)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(hsv)

            # cv2.imshow('jeje', equalized)
            mask = cv2.inRange(equalized, gray_min, gray_max)
            result = cv2.bitwise_and(par_image, par_image, mask=mask)

            # Display result image
            cv2.imshow('image', result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    # pylint: disable=too-many-locals
    def color_hsl_separator(self, par_image: ndarray) -> None:
        """
        Creates a window with trackbars for color change, and allows the user to separate an image 
            into different colors
        based on the HSL (Hue, Saturation, Lightness) values set by the trackbars.

        Parameters:
        par_image (numpy.ndarray): The image to be separated.

        Returns:
        None
        """
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
        ph_min = ps_min = pv_min = ph_max = ps_max = pv_max = 0

        while 1:
            # Get current positions of all trackbars
            h_min = cv2.getTrackbarPos('HMin', 'image')
            s_min = cv2.getTrackbarPos('SMin', 'image')
            v_min = cv2.getTrackbarPos('VMin', 'image')
            h_max = cv2.getTrackbarPos('HMax', 'image')
            s_max = cv2.getTrackbarPos('SMax', 'image')
            v_max = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and maximum HSV values to display
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(par_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(par_image, par_image, mask=mask)

            # Print if there is a change in HSV value
            if ((ph_min != h_min) | (ps_min != s_min) | (pv_min != v_min) | (ph_max != h_max) | (
                    ps_max != s_max) | (
                    pv_max != v_max)):
                print(f"(hMin = {h_min}, sMin = {s_min}, vMin = {v_min}),"
                      f" (hMax = {h_max}, sMax = {s_max}, vMax = {v_max})")
                ph_min = h_min
                ps_min = s_min
                pv_min = v_min
                ph_max = h_max
                ps_max = s_max
                pv_max = v_max

            # Display result image
            cv2.imshow('image', result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

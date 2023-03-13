"""
This module provides a set of tools for image manipulation, including loading images for comparison,
defining a region of interest in an image, and performing template matching using OpenCV. These 
tools can be used to automate tasks that involve processing or analyzing images, such as computer
vision and machine learning applications.
"""
import os
from typing import Dict

import cv2
import numpy as np
from numpy import ndarray


class ImageManipulation:
    """
        This class provides various methods for manipulating images using OpenCV, including
        loading comparable images for template matching, applying a region of interest mask to
        an image, and matching a template image with an original image using OpenCV's template
        matching.
    """

    comparable_images: Dict[str, ndarray] = None

    def __init__(self):
        pass

    def load_comparable_images(self) -> bool:
        """
        Loads Images which are used for comparison e.g. matchTemplate
        :return: Returns True if loaded correctly otherwise False
        """
        self.comparable_images: Dict[str, np.ndarray] = {}
        directory: str = "comparable_images"
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                img: np.ndarray = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_COLOR)
                name: str = os.path.splitext(filename)[0]
                self.comparable_images[name] = img
        return True

    def region_of_interest(self, par_img: ndarray, par_vertices):
        """
        Applies a region of interest mask to the input image.

        Args:
            par_img: The input image as a NumPy array.
            par_vertices: A list of vertices defining the region of interest polygon.

        Returns:
            A masked image as a NumPy array, where only the pixels inside the region of interest 
            are retained.
        """
        mask = np.zeros_like(par_img)
        # color_channel_count = img.shape[2]
        create_matched_color_mask = 255
        # Fills Polygons That Are Not For Our Interest
        cv2.fillPoly(mask, par_vertices, create_matched_color_mask)
        # Return Image Where Only The Mask Pixel Matches
        masked_image = cv2.bitwise_and(par_img, mask)
        return masked_image

    def match_template(self, par_original_image: ndarray, par_template_to_match: ndarray) -> bool:
        """
            Matches a template image with an original image using OpenCV's template matching.

            Args:
                par_original_image: A numpy array representing the original image.
                par_template_to_match: A numpy array representing the template image to match.

            Returns:
                A boolean value indicating whether the template image was found in the original
                 image.
        """
        # Convert to grayscale
        screen_gray = cv2.cvtColor(par_original_image, cv2.COLOR_BGR2GRAY)
        restart_state_gray = cv2.cvtColor(par_template_to_match, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        res = cv2.matchTemplate(screen_gray, restart_state_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        print(loc)

        # Return boolean based on whether match was found or not
        return len(loc[0]) > 0

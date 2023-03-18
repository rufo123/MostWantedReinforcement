"""
This is a module that provides class for managing font settings to be used in OpenCV.

Classes:
    FontSettings: A class for managing font settings.
"""
import cv2


class FontSettings:
    """
       A class for defining font settings to be used in OpenCV.
    """
    _a_font: int = cv2.FONT_HERSHEY_SIMPLEX
    _a_font_scale: float = 1
    _a_thickness: int = 2
    _a_line_type: int = 2

    def __init__(self,
                 par_font: int,
                 par_font_scale: float,
                 par_font_thickness: int,
                 par_line_type: int):
        """
        Initializes the font settings with given values.

        Args:
            par_font (int): The font type to be used.
            par_font_scale (float): The size of the font.
            par_font_thickness (int): The thickness of the font.
            par_line_type (int): The type of line to be used.
        """
        _a_font = par_font
        _a_font_scale = par_font_scale
        _a_thickness = par_font_thickness
        _a_line_type = par_line_type

    @property
    def font(self) -> int:
        """Returns the font type."""
        return self._a_font

    @property
    def font_scale(self) -> float:
        """Returns the font size."""
        return self._a_font_scale

    @property
    def thickness(self) -> int:
        """Returns the font thickness."""
        return self._a_thickness

    @property
    def line_type(self) -> int:
        """Returns the type of line to be used."""
        return self._a_line_type

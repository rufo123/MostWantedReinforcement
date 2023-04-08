"""
This module defines the `EnabledGameApiValues` class which provides information about the
 enabled/disabled values in the game API.
"""


class EnabledGameApiValues:
    """
    A class that represents a set of enabled game API values.

    Attributes:
        __enabled_car_speed_mph (bool): Indicates whether the car speed (in miles per hour)
         is enabled.
        __enabled_distance_offset_center (bool): Indicates whether the distance offset from
         the center is enabled.
        __enabled_distance_incline_center (bool): Indicates whether the distance incline from
         the center is enabled.
        __enabled_lap_progress (bool): Indicates whether the lap progress is enabled.
        __enabled_wrong_way_indicator (bool): Indicates whether the wrong way indicator is
         enabled.
        __enabled_revolutions_per_minute (bool): Indicates whether the revolutions per minute
         (RPM) is enabled.
        __enabled_mini_map (bool): Indicates whether the mini map is enabled.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 par_enabled_car_speed=False,
                 par_enabled_distance_offset_center=False,
                 par_enabled_distance_incline_center=False,
                 par_enabled_lap_progress=False,
                 par_enabled_wrong_way_indicator=False,
                 par_enabled_revolutions_per_minute=False,
                 par_enabled_mini_map=False
                 ):
        self.__enabled_car_speed_mph = par_enabled_car_speed
        self.__enabled_distance_offset_center = par_enabled_distance_offset_center
        self.__enabled_distance_incline_center = par_enabled_distance_incline_center
        self.__enabled_lap_progress = par_enabled_lap_progress
        self.__enabled_wrong_way_indicator = par_enabled_wrong_way_indicator
        self.__enabled_revolutions_per_minute = par_enabled_revolutions_per_minute
        self.__enabled_mini_map = par_enabled_mini_map

    @property
    def enabled_car_speed_mph(self) -> bool:
        """
        Returns the value of the `__enabled_car_speed_mph` attribute.
    
        Returns:
        --------
        bool
            The value of the `__enabled_car_speed_mph` attribute.
        """
        return self.__enabled_car_speed_mph

    @property
    def enabled_distance_offset_center(self) -> bool:
        """
        Returns the value of the `__enabled_distance_offset_center` attribute.
    
        Returns:
        --------
        bool
            The value of the `__enabled_distance_offset_center` attribute.
        """
        return self.__enabled_distance_offset_center

    @property
    def enabled_distance_incline_center(self) -> bool:
        """
        Returns the value of the `__enabled_distance_incline_center` attribute.
    
        Returns:
        --------
        bool
            The value of the `__enabled_distance_incline_center` attribute.
        """
        return self.__enabled_distance_incline_center

    @property
    def enabled_lap_progress(self) -> bool:
        """
        Returns the value of the `__enabled_lap_progress` attribute.
    
        Returns:
        --------
        bool
            The value of the `__enabled_lap_progress` attribute.
        """
        return self.__enabled_lap_progress

    @property
    def enabled_wrong_way_indicator(self) -> bool:
        """
        Returns the value of the `__enabled_wrong_way_indicator` attribute.
    
        Returns:
        --------
        bool
            The value of the `__enabled_wrong_way_indicator`
        """
        return self.__enabled_wrong_way_indicator

    @property
    def enabled_revolutions_per_minute(self) -> bool:
        """
        Returns the value of the `__enabled_revolutions_per_minute` attribute.

        Returns:
        --------
        bool
            The value of the `__enabled_revolutions_per_minute` attribute.
        """
        return self.__enabled_revolutions_per_minute

    @property
    def enabled_mini_map(self) -> bool:
        """
        Returns the value of the `__enabled_mini_map` attribute.

        Returns:
        --------
        bool
            The value of the `__enabled_mini_map` attribute.
        """
        return self.__enabled_mini_map

    @property
    def count_enabled_values(self) -> int:
        """
        Returns the count of enabled attributes/values.

        Returns:
        --------
        bool
            The count of enabled attributes/values.
        """
        return sum([
            self.__enabled_car_speed_mph,
            self.__enabled_distance_offset_center,
            self.__enabled_distance_incline_center,
            self.__enabled_lap_progress,
            self.__enabled_wrong_way_indicator,
            self.__enabled_revolutions_per_minute,
            self.__enabled_mini_map
        ])

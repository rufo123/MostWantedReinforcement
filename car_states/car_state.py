"""
The `car_states` module contains the `CarState` class which represents the state of a car
 in environment
"""


class CarState:
    """
    A class representing the state of a car in a racing game.

    Args:
        par_speed_mph (float): The current speed of the car in miles per hour.
        par_distance_offset_center (float): The distance of the car from the center of the track.
        par_lap_progress (float): The current progress of the car in the current lap, ranging
            from 0 to 100.
        par_incline_center (float): The current inclination of the road at the car's position.
        par_revolutions_per_minute (float): The current RPM of the car's engine.
        par_wrong_way_indicator (float): Whether the car is currently driving in the wrong 
            direction.

    """

    # pylint: disable=too-many-arguments
    def __init__(self, par_speed_mph=-1, par_distance_offset_center=-1, par_lap_progress=-1,
                 par_incline_center=-1, par_revolutions_per_minute=-1, par_wrong_way_indicator=-1):
        self._speed_mph = par_speed_mph
        self._distance_offset_center = par_distance_offset_center
        self._lap_progress = par_lap_progress
        self._incline_center = par_incline_center
        self._revolutions_per_minute = par_revolutions_per_minute
        self._wrong_way_indicator = par_wrong_way_indicator

    @property
    def speed_mph(self) -> float:
        """The current speed of the car in miles per hour."""
        return self._speed_mph

    @property
    def distance_offset_center(self) -> float:
        """The distance of the car from the center of the track."""
        return self._distance_offset_center

    @property
    def lap_progress(self) -> float:
        """The current progress of the car in the current lap, ranging from 0 to 1."""
        return self._lap_progress

    @property
    def incline_center(self) -> float:
        """The current inclination of the road at the car's position."""
        return self._incline_center

    @property
    def revolutions_per_minute(self) -> float:
        """The current RPM of the car's engine."""
        return self._revolutions_per_minute

    @property
    def wrong_way_indicator(self) -> float:
        """Whether the car is currently driving in the wrong direction."""
        return self._wrong_way_indicator

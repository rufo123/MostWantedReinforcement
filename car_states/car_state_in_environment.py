"""
A subclass of `CarState` that represents the state of a car in an environment.

This class extends `CarState` by adding an `lap_progress_difference` property that represents
 the difference of car progress between previous and current step from environment.
"""
from car_states.car_state import CarState


# pylint: disable=too-few-public-methods
class CarStateInEnvironment(CarState):
    """
    Represents the state of a car in an environment.

    Inherits from the CarState class, and adds an `lap_progress_difference` property to represent
    the difference of car progress between previous and current step from environment.

    Args:
        par_lap_progress_difference (float): The difference of car progress between previous
         and current step
        par_speed_mph (float): The car's speed in miles per hour.
        par_distance_offset_center (float): The distance between the center of the car
            and the center of the road, in meters.
        par_lap_progress (float): The car's progress around the track, expressed as a
            percentage of the track completed.
        par_incline_center (float): The angle of the road incline, in radians.
        par_revolutions_per_minute (float): The car's revolutions per minute (RPM).
        par_wrong_way_indicator (float): Indicates whether the car is currently driving
            the wrong way around the track (1.0) or not (0.0).
    """
    _a_lap_progress_difference: float

    # pylint: disable=too-many-arguments
    def __init__(self, par_lap_progress_difference: float,
                 par_speed_mph=-1, par_distance_offset_center=-1,
                 par_lap_progress=-1, par_incline_center=-1, par_revolutions_per_minute=-1,
                 par_wrong_way_indicator=-1):
        self._a_action = par_lap_progress_difference
        super().__init__(par_speed_mph=par_speed_mph,
                         par_distance_offset_center=par_distance_offset_center,
                         par_lap_progress=par_lap_progress,
                         par_incline_center=par_incline_center,
                         par_revolutions_per_minute=par_revolutions_per_minute,
                         par_wrong_way_indicator=par_wrong_way_indicator)

    @property
    def lap_progress_difference(self) -> float:
        """The difference of car progress between previous and current step."""
        return self._a_action

"""
A subclass of `CarState` that represents the state of a car in an environment.

This class extends `CarState` by adding an `lap_progress_difference` property that represents
 the difference of car progress between previous and current step from environment.
"""
import numpy

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
        par_mini_map (numpy.ndarray): A 3D array representing a top-down view of the track
            and the car's position. Defaults to None.
        par_car_state (CarState): An instance of the parent class `CarState` to copy values from.
            Defaults to None.
    """
    _a_lap_progress_difference: float

    # pylint: disable=too-many-arguments
    def __init__(self, par_lap_progress_difference: float = -1,
                 par_speed_mph=-1, par_distance_offset_center=-1,
                 par_lap_progress=-1, par_incline_center=-1, par_revolutions_per_minute=-1,
                 par_wrong_way_indicator=-1, par_mini_map=None, par_car_state: CarState = None):
        self._a_lap_progress_difference = par_lap_progress_difference

        if par_car_state:
            super().__init__(par_speed_mph=par_car_state.speed_mph,
                             par_distance_offset_center=par_car_state.distance_offset_center,
                             par_lap_progress=par_car_state.lap_progress,
                             par_incline_center=par_car_state.incline_center,
                             par_revolutions_per_minute=par_car_state.revolutions_per_minute,
                             par_wrong_way_indicator=par_car_state.wrong_way_indicator,
                             par_mini_map=par_car_state.mini_map)
        else:
            super().__init__(par_speed_mph=par_speed_mph,
                             par_distance_offset_center=par_distance_offset_center,
                             par_lap_progress=par_lap_progress,
                             par_incline_center=par_incline_center,
                             par_revolutions_per_minute=par_revolutions_per_minute,
                             par_wrong_way_indicator=par_wrong_way_indicator,
                             par_mini_map=par_mini_map)

    # pylint: disable=arguments-renamed
    def assign_values(self, par_lap_progress_difference: float = -1,
                      par_speed_mph=-1, par_distance_offset_center=-1, par_lap_progress=-1,
                      par_incline_center=-1, par_revolutions_per_minute=-1,
                      par_wrong_way_indicator=-1,
                      par_mini_map: numpy.ndarray = None,
                      par_car_state: CarState = None):
        """
        Assign values to the attributes of an object.

        Parameters:
            par_lap_progress_difference (float): The difference of car progress between previous
         and current step
            par_speed_mph (float): The speed of the car in mph.
            par_distance_offset_center (float): The distance of the car from the center of the
             track.
            par_lap_progress (float): The progress of the car in the current lap.
            par_incline_center (float): The inclination of the track at the current position of the
             car.
            par_revolutions_per_minute (float): The number of revolutions of the car's engine per
             minute.
            par_wrong_way_indicator (int): A binary indicator (0 or 1) that shows if the car is
             going the wrong way.
            par_mini_map (numpy.ndarray): A 2D array that represents the mini-map of the track.
            par_car_state (CarState): An instance of the parent class `CarState` to copy values
             from. Defaults to None.

        Returns:
            None
        """
        self._a_lap_progress_difference = par_lap_progress_difference

        if par_car_state:
            super().assign_values(
                par_speed_mph=par_car_state.speed_mph,
                par_distance_offset_center=par_car_state.distance_offset_center,
                par_lap_progress=par_car_state.lap_progress,
                par_incline_center=par_car_state.incline_center,
                par_revolutions_per_minute=par_car_state.revolutions_per_minute,
                par_wrong_way_indicator=par_car_state.wrong_way_indicator,
                par_mini_map=par_car_state.mini_map
            )
        else:
            super().assign_values(
                par_speed_mph=par_speed_mph,
                par_distance_offset_center=par_distance_offset_center,
                par_lap_progress=par_lap_progress,
                par_incline_center=par_incline_center,
                par_revolutions_per_minute=par_revolutions_per_minute,
                par_wrong_way_indicator=par_wrong_way_indicator,
                par_mini_map=par_mini_map
            )

    def reset_car_state(self) -> None:
        """
        Resets the values of the attributes of the object to their initial state.

        Returns:
            None
        """
        self._a_lap_progress_difference = -1
        super().reset_car_state()

    @property
    def lap_progress_difference(self) -> float:
        """The difference of car progress between previous and current step."""
        return self._a_lap_progress_difference

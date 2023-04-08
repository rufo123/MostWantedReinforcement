"""
Module for implementing a state calculation strategy that stores a 5x5 matrix of past states
 and actions and returns it as a normalized flattened Torch Tensor.

This module defines the `BiggerStateStrategy` class, which inherits from the abstract base class
 `StateCalculationStrategy`.
This class overrides the `calculate_state` method to store the last
 five observations and actions in a matrix, which is then flattened and returned as a Torch Tensor.
"""
from typing import Union

import numpy
import torch

from car_states.car_state_in_environment import CarStateInEnvironment
from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy


# pylint: disable=R0801
class MinimapStateStrategy(AStateCalculationStrategy):
    """
    A state calculation strategy that stores the last five observations and actions in a 5x5 matrix
     and returns it as a normalized flattened Torch Tensor.

    This class inherits from the abstract base class `StateCalculationStrategy`. It overrides the
     `calculate_state` method to store the last five observations and actions in a 5x5 matrix. The
     matrix is then flattened and returned as a Torch Tensor.

    Attributes:
        a_state_matrix (np.ndarray): A 5x5 matrix that stores the last five observations and
         actions.
    """

    def calculate_state(self, par_car_state: CarStateInEnvironment,
                        par_action_taken: Union[int, None]) -> torch.Tensor:
        """
        Calculate the state tensor for a given car state and action taken.

        Args:
            par_car_state (CarStateInEnvironment): The car state in the environment.
            par_action_taken (Union[int, None]): The action taken by the car, or None if no 
             has been taken.

        Returns:
            torch.Tensor: A 3D tensor of shape (4, 48, 48) representing the current state of the
             car.
            The four layers of the tensor represent the mini-map, lap progress, car speed,
               and wrong-way indicator, respectively.
            The values in the mini-map layer are resized to 48x48 pixels and the lap progress and
             car speed layers are rounded to a specified number of digits.
            The wrong-way indicator layer contains binary values indicating whether the car is going
             the wrong way on the track or not.
        """
        normalized_state_values: CarStateInEnvironment = self.normalize_state_values(par_car_state)

        mini_map_resized_2d = numpy.resize(par_car_state.mini_map, (48, 48))
        lap_progress_2d = numpy.full((48, 48),
                                     round(normalized_state_values.lap_progress, ndigits=5), )
        car_speed_2d = numpy.full((48, 48),
                                  round(normalized_state_values.speed_mph, ndigits=6), )
        wrong_way_2d = numpy.full((48, 48), par_car_state.wrong_way_indicator)

        new_state = numpy.stack((mini_map_resized_2d, lap_progress_2d, car_speed_2d, wrong_way_2d))

        # returns 3D matrix of size 4x48x48
        return torch.from_numpy(new_state)

    def normalize_state_values(self, par_car_state_not_normalized: CarStateInEnvironment) \
            -> CarStateInEnvironment:
        """
        Normalizes the input state values and returns a tuple of normalized values.
        Args:
            par_car_state_not_normalized (CarState): A CarState object of un-normalized state
                values, including the car speed, distance offset, lap progress, and direction
                offset.
        Returns:
            CarState: A CarState object of normalized state values, including the
                normalized car speed, normalized distance offset, normalized lap progress,
                and normalized direction offset.
        """
        tmp_car_top_speed: float = 111
        tmp_normalized_speed = par_car_state_not_normalized.speed_mph / tmp_car_top_speed

        if par_car_state_not_normalized.distance_offset_center >= 0:
            tmp_normalized_distance_offset: float = 1
        elif par_car_state_not_normalized.distance_offset_center >= -1:
            tmp_normalized_distance_offset: float = \
                1 + par_car_state_not_normalized.distance_offset_center
        elif par_car_state_not_normalized.distance_offset_center >= -50:
            tmp_normalized_distance_offset: float = \
                (1 + par_car_state_not_normalized.distance_offset_center) / 49
        else:
            tmp_normalized_distance_offset: float = -1

        tmp_normalized_lap_progress: float = par_car_state_not_normalized.lap_progress / 100

        tmp_normalized_direction_offset: float = par_car_state_not_normalized.incline_center

        return CarStateInEnvironment(
            par_speed_mph=round(tmp_normalized_speed, ndigits=6),
            par_distance_offset_center=round(tmp_normalized_distance_offset, ndigits=6),
            par_lap_progress=round(tmp_normalized_lap_progress, ndigits=5),
            par_incline_center=tmp_normalized_direction_offset
        )

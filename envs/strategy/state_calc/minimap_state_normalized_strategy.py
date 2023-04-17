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
from numpy import ndarray

from car_states.car_state_in_environment import CarStateInEnvironment
from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy


# pylint: disable=R0801
# noinspection DuplicatedCode
class MinimapStateNormalizedStrategy(AStateCalculationStrategy):
    """
    A state calculation strategy that stores the last five observations and actions in a 5x5 matrix
     and returns it as a normalized flattened Torch Tensor.

    This class inherits from the abstract base class `StateCalculationStrategy`. It overrides the
     `calculate_state` method to store the last five observations and actions in a 5x5 matrix. The
     matrix is then flattened and returned as a Torch Tensor.

    Attributes:
        a_car_state_in_environment (np.ndarray): A object represent car state in environment
    """

    a_car_state_in_environment: CarStateInEnvironment

    def __init__(self):
        self.a_car_state_in_environment = CarStateInEnvironment()

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

        mini_map_resized_2d = normalized_state_values.mini_map
        lap_progress_2d = numpy.full((48, 48), normalized_state_values.lap_progress)
        car_speed_2d = numpy.full((48, 48), normalized_state_values.speed_mph)
        wrong_way_2d = numpy.full((48, 48), normalized_state_values.wrong_way_indicator)

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

        self.a_car_state_in_environment.reset_car_state()

        tmp_normalized_minimap: ndarray = \
            par_car_state_not_normalized.mini_map.astype('float32') / 255.0

        tmp_car_top_speed: float = 111
        tmp_normalized_speed = par_car_state_not_normalized.speed_mph / tmp_car_top_speed

        tmp_normalized_lap_progress: float = par_car_state_not_normalized.lap_progress / 100

        self.a_car_state_in_environment.reset_car_state()
        self.a_car_state_in_environment.assign_values(
            par_speed_mph=tmp_normalized_speed,
            par_lap_progress=tmp_normalized_lap_progress,
            par_wrong_way_indicator=par_car_state_not_normalized.wrong_way_indicator,
            par_mini_map=tmp_normalized_minimap
        )
        return self.a_car_state_in_environment

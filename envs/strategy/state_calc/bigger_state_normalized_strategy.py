"""
Module for implementing a state calculation strategy that stores a 5x5 matrix of past states
 and actions and returns it as a normalized flattened Torch Tensor.

This module defines the `BiggerStateStrategy` class, which inherits from the abstract base class
 `StateCalculationStrategy`.
This class overrides the `calculate_state` method to store the last
 five observations and actions in a matrix, which is then flattened and returned as a Torch Tensor.
"""
from typing import Union

import numpy as np
import torch

from car_states.car_state_in_environment import CarStateInEnvironment
from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy


# pylint: disable=R0801
class BiggerStateNormalizedStrategy(AStateCalculationStrategy):
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
    a_state_matrix: np.ndarray

    def __init__(self):
        """
        Initializes the BiggerStateNormalizedStrategy object by creating a 5x5 matrix filled with
         -1.
        """
        self.a_state_matrix = np.zeros((5, 5), dtype=float) - 1

    def calculate_state(self, par_car_state: CarStateInEnvironment,
                        par_action_taken: Union[int, None]) -> torch.Tensor:
        """
        Shifts the rows of the state matrix down by one row and inserts the new input parameters
        in the first row. Prints the updated matrix.
        Args:
            par_action_taken: The current action by the car from th environment [ACTION_TAKEN]
            par_car_state: A object of CarState class representing the current
                car state: 
                    [CAR_SPEED, DISTANCE_FROM_CENTER, LAP_PROGRESS, INCLINE_FROM_CENTER]
        Returns:
            Torch Tensor representing state as a tensor
        """

        normalized_state_values: CarStateInEnvironment = self.normalize_state_values(par_car_state)

        current_inputs_rounded: tuple[int, float, float, float, float] = (
            par_action_taken,
            round(normalized_state_values.speed_mph, ndigits=6),
            round(normalized_state_values.distance_offset_center, ndigits=6),
            round(normalized_state_values.lap_progress, ndigits=5),
            normalized_state_values.incline_center
        )

        # shift the rows down
        self.a_state_matrix[1:, :] = self.a_state_matrix[:-1, :]
        # insert the new parameters in the first row
        self.a_state_matrix[0, :] = current_inputs_rounded
        # print the updated matrix
        # print("ACTION, CAR_SPEED, DISTANCE_FROM_CENTER, LAP_PROGRESS, INCLINE_FROM_CENTER")
        # print(self.a_state_matrix)

        return torch.tensor(self.a_state_matrix.flatten()).view(1, 1, 25)

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

"""
Module for implementing a state calculation strategy that stores a 5x5 matrix of past states
 and actions and returns it as a flattened Torch Tensor.

This module defines the `BiggerStateStrategy` class, which inherits from the abstract base class
 `StateCalculationStrategy`.
This class overrides the `calculate_state` method to store the last
 five observations and actions in a matrix, which is then flattened and returned as a Torch Tensor.
"""
import numpy as np
import torch

from car_states.car_state_in_environment import CarStateInEnvironment
from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy


# pylint: disable=too-few-public-methods
# pylint: disable=R0801
class BiggerStateStrategy(AStateCalculationStrategy):
    """
    A state calculation strategy that stores the last five observations and actions in a 5x5 matrix
     and returns it as a flattened Torch Tensor.

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
        Initializes the BiggerStateStrategy object by creating a 5x5 matrix filled with -1.
        """
        self.a_state_matrix = np.zeros((5, 5), dtype=float) - 1

    def calculate_state(self, par_car_state: CarStateInEnvironment,
                        par_action_taken: int) -> torch.Tensor:
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
        current_inputs_rounded: tuple[int, float, float, float, float] = (
            par_action_taken,
            par_car_state.speed_mph,
            round(par_car_state.distance_offset_center, ndigits=6),
            par_car_state.lap_progress,
            par_car_state.incline_center
        )
        # shift the rows down
        self.a_state_matrix[1:, :] = self.a_state_matrix[:-1, :]
        # insert the new parameters in the first row
        self.a_state_matrix[0, :] = current_inputs_rounded
        # print_utils the updated matrix
        # print_utils("ACTION, CAR_SPEED, DISTANCE_FROM_CENTER, LAP_PROGRESS, INCLINE_FROM_CENTER")
        # print_utils(self.a_state_matrix)

        return torch.tensor(self.a_state_matrix.flatten()).view(1, 1, 25)

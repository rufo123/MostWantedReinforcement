"""
Module for implementing strategies for basic state calculation in a reinforcement learning
 environment.

The module contains a class BasicStateStrategy which implements the StateCalculationStrategy
 abstract
class to provide a basic strategy for calculating the state of the environment. The strategy
 calculates
the state based on the speed of the car, its distance offset from the center of the track, its lap
 progress,
and its distance offset from the center of the track.
"""

import torch

from car_states.car_state import CarState
from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy


# pylint: disable=too-few-public-methods
# pylint: disable=R0801
class BasicStateStrategy(AStateCalculationStrategy):
    """
    A basic state calculation strategy for a reinforcement learning environment.

    This class implements the `StateCalculationStrategy` abstract class to provide a basic strategy
     for calculating the state of the environment. The strategy calculates the state based on
     the speed of the car, its distance offset from the center of the track, its lap progress
     and its distance offset from the center
     of the track.
    """

    # pylint-disable=unused-argument
    def calculate_state(self, par_car_state: CarState, par_action_taken=None) -> torch.Tensor:
        """
        Calculates the state of the environment based on the car state.

        Args:
            par_car_state: The car state.
            par_action_taken: 

        Returns:
            The calculated state as a tensor.
            
        """
        return torch.tensor([
            par_car_state.speed_mph,
            par_car_state.distance_offset_center,
            par_car_state.lap_progress,
            par_car_state.distance_offset_center
        ])

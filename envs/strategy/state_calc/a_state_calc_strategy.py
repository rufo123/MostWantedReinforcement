"""
Module for implementing different strategies for calculating the state in a reinforcement learning
 environment.
"""
from abc import ABC, abstractmethod
from typing import Union

import numpy
import torch

from car_states.car_state import CarState
from car_states.car_state_in_environment import CarStateInEnvironment


# pylint: disable=too-few-public-methods
class AStateCalculationStrategy(ABC):
    """
    Abstract base class for state calculation strategies.
    """

    @abstractmethod
    def calculate_state(self,
                        par_car_state: [CarState, CarStateInEnvironment],
                        par_action_taken: Union[int, None]
                        ) -> Union[torch.Tensor, numpy.ndarray]:
        """
        Calculates the state of the environment based on the observation.

        Args:
            par_car_state: A car state from the environment.
            par_action_taken: The current action by the car from th environment [ACTION_TAKEN]

        Returns:
            The calculated state.
        """

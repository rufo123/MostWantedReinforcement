"""
The ARewardStrategy module defines an abstract class for reward strategies to be used in 
    reinforcement learning
environments. It defines an interface for reward strategy classes to implement. 
"""
from abc import ABC, abstractmethod

from car_states.car_state_in_environment import CarStateInEnvironment


# pylint: disable=too-few-public-methods
class ARewardStrategy(ABC):
    """
    Abstract base class for implementing reward strategies in a reinforcement learning environment.
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate_reward(self, par_env_inputs: CarStateInEnvironment,
                        par_game_steps_per_episode: int,
                        par_env_steps_counter: int,
                        par_terminal: bool) -> tuple[float, bool]:
        """
        Method to evaluate the reward of a given state and environment.

        Args:
            par_env_inputs (CarStateInEnvironment): Object containing car state represented by 
                the environment
            par_game_steps_per_episode (int): Count of Configured Game Steps per Env Episode
            par_env_steps_counter: (int) Count of passed game Steps in Env
            par_terminal (bool): A flag indicating whether the current state is terminal.

        Returns:
            Tuple[float, bool]: A tuple containing the reward and a flag indicating whether the
                current state is terminal.
        """

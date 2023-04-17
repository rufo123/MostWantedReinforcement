"""
Module: This module defines the interface for configuration of a short race factory.
"""
from abc import ABC, abstractmethod

from car_states.enabled_game_api_values import EnabledGameApiValues
from envs.strategy.reward.a_reward_strategy import ARewardStrategy
from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy
from models.abstract_factory.a_short_race_factory import AShortRaceFactory


class IConfiguration(ABC):
    """
    Interface for configuration of a short race factory.
    """

    @abstractmethod
    def return_model(self) -> AShortRaceFactory:
        """
        Return an instance of AShortRaceFactory.


        Returns:
            AShortRaceFactory: An instance of AShortRaceFactory.
        """

    @abstractmethod
    def return_reward_strategy(self, par_iteration_number=0) -> ARewardStrategy:
        """
        Return an instance of ARewardStrategy.

        Returns:
            ARewardStrategy: An instance of ARewardStrategy.
            par_iteration_number: A optional value to set dynamic reward strategies based on
                current iteration number
        """

    @abstractmethod
    def return_state_calc_strategy(self, par_iteration_number=0) -> AStateCalculationStrategy:
        """
        Return an instance of AStateCalculationStrategy.

        Returns:
            AStateCalculationStrategy: An instance of AStateCalculationStrategy.
            par_iteration_number: A optional value to set dynamic state calc strategies based on
                current iteration number
        """

    @abstractmethod
    def return_enabled_game_api_values(self) -> EnabledGameApiValues:
        """
        Return an instance of EnabledGameApiValues.

        Returns:
            EnabledGameApiValues: An instance of EnabledGameApiValues.
        """

    @abstractmethod
    def return_name(self) -> str:
        """
          Return a name of the experiment (used mainly for folder naming).
    
          Returns:
              str: a name of the experiment.
        """

    @abstractmethod
    def return_dimensional_input(self) -> tuple:
        """
        Return a dimensional input for the experiment.
    
          Returns:
              tuple: a dimensional input for the experiment.
        """

    @abstractmethod
    def return_max_speed_non_visualised(self) -> int:
        """
        Return the maximum speed for the non-visualized experiment.

        Returns:
            int: The maximum speed for the non-visualized experiment.
        """

    @abstractmethod
    def return_max_speed_visualised(self) -> int:
        """
        Return the maximum speed for the visualized experiment.

        Returns:
            int: The maximum speed for the visualized experiment.
        """

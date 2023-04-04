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
    def return_reward_strategy(self) -> ARewardStrategy:
        """
        Return an instance of ARewardStrategy.

        Returns:
            ARewardStrategy: An instance of ARewardStrategy.
        """

    @abstractmethod
    def return_state_calc_strategy(self) -> AStateCalculationStrategy:
        """
        Return an instance of AStateCalculationStrategy.

        Returns:
            AStateCalculationStrategy: An instance of AStateCalculationStrategy.
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

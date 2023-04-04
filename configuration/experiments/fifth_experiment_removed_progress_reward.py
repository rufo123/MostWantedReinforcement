"""
This module contains the `FifthExperimentRemovedProgressReward` class, which implements the
 `IConfiguration` interface to define the model, reward strategy, and state calculation strategy
  to be used in the fourth experiment. 
 In this experiment, in addition to scaling of terminal function, step reward for lap progress is
  removed
"""
from car_states.enabled_game_api_values import EnabledGameApiValues
from configuration.factory.configuration_factory import ConfigurationFactory
from configuration.i_configuration import IConfiguration
from envs.strategy.reward.a_reward_strategy import ARewardStrategy
from envs.strategy.reward.reward_strategy_enum import RewardStrategyEnum
from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy
from envs.strategy.state_calc.state_strategy_enum import StateStrategyEnum
from models.abstract_factory.a_short_race_factory import AShortRaceFactory
from models.abstract_factory.short_race_factory_forward import ShortRaceFactoryForward


# pylint: disable=R0801
class FifthExperimentRemovedProgressReward(IConfiguration):
    """
    A class used to define the model, reward strategy, and state calculation strategy to be used
     in the fifth experiment.

    Attributes:
        a_configuration_factory (ConfigurationFactory): An instance of `ConfigurationFactory` used
         to create the necessary objects.
    """

    def __init__(self):
        self.a_configuration_factory = ConfigurationFactory()

    def return_model(self) -> AShortRaceFactory:
        """
        Return a model object for the fifth experiment.

        Returns:
            AShortRaceFactory: A model object for the fifth experiment.
        """
        return self.a_configuration_factory.create_model(
            ShortRaceFactoryForward(
                par_count_of_inputs=25,
                par_count_of_actions=8
            )
        )

    def return_reward_strategy(self) -> ARewardStrategy:
        """
        Return a reward strategy object for the fifth experiment.

        Returns:
            ARewardStrategy: A reward strategy object for the fifth experiment.
        """
        return self.a_configuration_factory.create_reward_strategy(
            RewardStrategyEnum.THIRD_REWARD_STRATEGY
        )

    def return_state_calc_strategy(self) -> AStateCalculationStrategy:
        """
        Return a state calculation strategy object for the fifth experiment.

        Returns:
            AStateCalculationStrategy: A state calculation strategy object for the fifth experiment.
        """
        return self.a_configuration_factory.create_state_calc_strategy(
            StateStrategyEnum.BIGGER_STATE_STRATEGY_NORMALIZED
        )

    def return_enabled_game_api_values(self) -> EnabledGameApiValues:
        """
        Returns an instance of `EnabledGameApiValues` that is used to enable specific values
        provided by GameApi for the sixth experiment, disabled values will not be available,
         to save processing power.
        Enabled: offset_center, incline_center, car_speed, lap_progress

        Returns:
            EnabledGameApiValues: An enabled game api values for the fifth experiment.
        """
        return self.a_configuration_factory.create_enabled_game_api_values(
            par_enabled_distance_offset_center=True,
            par_enabled_distance_incline_center=True,
            par_enabled_car_speed=True,
            par_enabled_lap_progress=True,
        )

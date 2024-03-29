"""
Module containing the abstract factory class for creating models for a reinforcement learning agent
 to play Need for Speed Most Wanted (2005).
"""
from typing import Union

from car_states.enabled_game_api_values import EnabledGameApiValues
from envs.strategy.reward.a_reward_strategy import ARewardStrategy
from envs.strategy.reward.reward_strategy_enum import RewardStrategyEnum
from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy
from envs.strategy.state_calc.state_strategy_enum import StateStrategyEnum
from models.abstract_factory.a_short_race_factory import AShortRaceFactory


class ConfigurationFactory:
    """
    Abstract factory class for creating models for a reinforcement learning agent
    to play Need for Speed Most Wanted (2005).
    """

    def create_model(self, par_model: AShortRaceFactory) -> AShortRaceFactory:
        """
        Creates a model for the reinforcement learning agent.

        :param par_model: A factory object that creates a model.
        :return: A factory object that creates a model.
        """
        return par_model

    def create_reward_strategy(self, par_reward_strategy: RewardStrategyEnum) -> ARewardStrategy:
        """
        Creates a reward strategy for the reinforcement learning agent.

        :param par_reward_strategy: An enumeration value that specifies the reward strategy.
        :return: An object that implements the reward strategy.
        """
        return par_reward_strategy.return_strategy()

    def create_dynamic_reward_strategy(self,
                                       par_border_iteration_numbers: list[int],
                                       par_reward_strategies: list[RewardStrategyEnum],
                                       par_iteration_number: int = 0):
        """
        Creates a dynamic reward strategy for the reinforcement learning agent based on the
         position of iteration number
        in the par_border_iteration_numbers list.

        Args:
        - par_border_iteration_numbers (list[int]): A list of border iteration numbers.
        - par_reward_strategies (list[RewardStrategyEnum]): A list of reward strategies.

        Returns:
        - ARewardStrategy: An object that implements the dynamic reward strategy.
        """
        # Check if the lengths of the input lists match
        if len(par_border_iteration_numbers) != len(par_reward_strategies):
            raise ValueError(
                "Length of border iteration numbers list does"
                " not match length of reward strategies list.")

        # Get the current iteration number
        current_iteration = par_iteration_number

        # Iterate through the border iteration numbers list and compare with current iteration
        # number
        for border_iteration, reward_strategy in zip(par_border_iteration_numbers,
                                                     par_reward_strategies):
            if current_iteration < border_iteration:
                return reward_strategy.return_strategy()

        # If current iteration number is greater than all border iteration numbers,
        # return the last reward strategy
        return par_reward_strategies[-1].return_strategy()

    def create_state_calc_strategy(
            self,
            par_state_calc_strategy: StateStrategyEnum
    ) -> AStateCalculationStrategy:
        """
        Creates a state calculation strategy for the reinforcement learning agent.

        :param par_state_calc_strategy: An enumeration value that specifies the state calculation
         strategy.
        :return: An object that implements the state calculation strategy.
        """
        return par_state_calc_strategy.return_strategy()

    def create_dynamic_state_calc_strategy(self,
                                           par_border_iteration_numbers: list[int],
                                           par_state_calc_strategies: list[StateStrategyEnum],
                                           par_iteration_number: int = 0):
        """
        Creates a dynamic state clac strategy for the reinforcement learning agent based on the
         position of iteration number
        in the par_state_calc_strategies list.

        Args:
        - par_border_iteration_numbers (list[int]): A list of border iteration numbers.
        - par_state_calc_strategies (list[StateStrategyEnum]): A list of state calc strategies.

        Returns:
        - ARewardStrategy: An object that implements the dynamic reward strategy.
        """
        # Check if the lengths of the input lists match
        if len(par_border_iteration_numbers) != len(par_state_calc_strategies):
            raise ValueError(
                "Length of border iteration numbers list does"
                " not match length of reward strategies list.")

        # Get the current iteration number
        current_iteration = par_iteration_number

        # Iterate through the border iteration numbers list and compare with current iteration
        # number
        for border_iteration, reward_strategy in zip(par_border_iteration_numbers,
                                                     par_state_calc_strategies):
            if current_iteration < border_iteration:
                return reward_strategy.return_strategy()

        # If current iteration number is greater than all border iteration numbers,
        # return the last reward strategy
        return par_state_calc_strategies[-1].return_strategy()

    def create_dimensional_input(self, par_input_dim: Union[tuple, int] = 4) -> tuple:
        """
        Creates a dimensional input tuple.

        Args:
            par_input_dim (Union[tuple, int]): The input dimensionality. If a tuple, returns it as
             is. 
                If an integer, returns a tuple with the integer as its only element. Default: 4.

        Returns:
            tuple: A tuple representing the input dimensionality.

        Raises:
            None.
        """
        if isinstance(par_input_dim, tuple):
            return par_input_dim
        return (par_input_dim,)

    # pylint: disable= too-many-arguments
    def create_enabled_game_api_values(
            self,
            par_enabled_car_speed=False,
            par_enabled_distance_offset_center=False,
            par_enabled_distance_incline_center=False,
            par_enabled_lap_progress=False,
            par_enabled_wrong_way_indicator=False,
            par_enabled_revolutions_per_minute=False,
            par_enabled_mini_map=False,
    ) -> EnabledGameApiValues:
        """
        Creates enabled game api values object for the reinforcement learning agent.

        :param par_enabled_car_speed: if sending of car speed in mph should be enabled.
        :param par_enabled_distance_offset_center: if sending of car offset from road center should
         be enabled.
        :param par_enabled_distance_incline_center: if sending of car incline from road center be
         enabled.
        :param par_enabled_lap_progress: if sending of lap progress should be enabled.
        :param par_enabled_wrong_way_indicator: if sending of wrong way indicator should be enabled.
        :param par_enabled_revolutions_per_minute: if sending of car's engine's revolutions per
         minute should be enabled.
        :param par_enabled_mini_map: if sending of grayscale minimap should be enabled
        :return: An object that describes enabled game api values.
        """
        return EnabledGameApiValues(
            par_enabled_car_speed=par_enabled_car_speed,
            par_enabled_distance_offset_center=par_enabled_distance_offset_center,
            par_enabled_distance_incline_center=par_enabled_distance_incline_center,
            par_enabled_lap_progress=par_enabled_lap_progress,
            par_enabled_wrong_way_indicator=par_enabled_wrong_way_indicator,
            par_enabled_revolutions_per_minute=par_enabled_revolutions_per_minute,
            par_enabled_mini_map=par_enabled_mini_map
        )

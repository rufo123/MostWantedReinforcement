"""
This module provides an enumeration of reward strategies and a function to return the
    appropriate reward strategy object.

"""
from enum import Enum

from envs.strategy.a_reward_strategy import ARewardStrategy
from envs.strategy.first_reward_strategy import FirstRewardStrategy
from envs.strategy.second_reward_strategy import SecondRewardStrategy


class RewardStrategyEnum(Enum):
    """
    Enumeration of available reward strategies.
    """
    FIRST_REWARD_STRATEGY = 0
    SECOND_REWARD_STRATEGY = 1

    def return_strategy(self) -> ARewardStrategy:
        """
        Returns an ARewardStrategy object based on the selected reward strategy enumeration.

        Args:
            self: The selected reward strategy enumeration.

        Returns:
            An ARewardStrategy object corresponding to the selected reward strategy enumeration.
             returns FirstRewardStrategy by default

        """
        # A dictionary that maps each enumeration to its corresponding reward strategy object.
        # It is slightly performance inefficient, because it should be class/instance level variable
        # but withing Enum class it is not possible to do.
        strategy_dict: dict[RewardStrategyEnum, ARewardStrategy] = {
            self.FIRST_REWARD_STRATEGY: FirstRewardStrategy(),
            self.SECOND_REWARD_STRATEGY: SecondRewardStrategy()
        }

        try:
            return strategy_dict[self]
        except KeyError:
            return FirstRewardStrategy()

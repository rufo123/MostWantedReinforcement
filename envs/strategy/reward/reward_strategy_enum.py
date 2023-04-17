"""
This module provides an enumeration of reward strategies and a function to return the
    appropriate reward strategy object.

"""
from enum import Enum

from envs.strategy.reward.a_reward_strategy import ARewardStrategy
from envs.strategy.reward.first_reward_strategy import FirstRewardStrategy
from envs.strategy.reward.fouth_minimap.fourth_minimap_10_percent_strateg import \
    FourthMinimap10PercentRewardStrategy
from envs.strategy.reward.fouth_minimap.fourth_minimap_20_percent_strategy import \
    FourthMinimap20PercentRewardStrategy
from envs.strategy.reward.fouth_minimap.fourth_minimap_40_percent_strategy import \
    FourthMinimap40PercentRewardStrategy
from envs.strategy.reward.second_reward_strategy import SecondRewardStrategy
from envs.strategy.reward.third_reward_strategy import ThirdRewardStrategy


class RewardStrategyEnum(Enum):
    """
    Enumeration of available reward strategies.
    """
    FIRST_REWARD_STRATEGY = 0
    SECOND_REWARD_STRATEGY = 1
    THIRD_REWARD_STRATEGY = 2
    FOURTH_MINIMAP_10_PERCENT = 3
    FOURTH_MINIMAP_20_PERCENT = 4
    FOURTH_MINIMAP_40_PERCENT = 5

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
            self.SECOND_REWARD_STRATEGY: SecondRewardStrategy(),
            self.THIRD_REWARD_STRATEGY: ThirdRewardStrategy(),
            self.FOURTH_MINIMAP_10_PERCENT: FourthMinimap10PercentRewardStrategy(),
            self.FOURTH_MINIMAP_20_PERCENT: FourthMinimap20PercentRewardStrategy(),
            self.FOURTH_MINIMAP_40_PERCENT: FourthMinimap40PercentRewardStrategy(),
        }

        try:
            return strategy_dict[self]
        except KeyError:
            return FirstRewardStrategy()

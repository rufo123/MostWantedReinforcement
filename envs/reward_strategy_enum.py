"""
This module provides an enumeration of reward strategies and a function to return the
    appropriate reward strategy object.

"""
from enum import Enum

from envs.a_reward_strategy import ARewardStrategy
from envs.first_reward_strategy import FirstRewardStrategy


class RewardStrategyEnum(Enum):
    """
    Enumeration of available reward strategies.
    """
    FIRST_REWARD_STRATEGY = 0

    # pylint: disable=no-else-return
    def return_strategy(self) -> ARewardStrategy:
        """
        Returns an ARewardStrategy object based on the selected reward strategy enumeration.

        Args:
            self: The selected reward strategy enumeration.

        Returns:
            An ARewardStrategy object corresponding to the selected reward strategy enumeration.

        """
        if self == RewardStrategyEnum.FIRST_REWARD_STRATEGY:
            return FirstRewardStrategy()
        else:
            return FirstRewardStrategy()

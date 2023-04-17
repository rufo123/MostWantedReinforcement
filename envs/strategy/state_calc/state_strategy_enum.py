"""
This module provides an enumeration of reward strategies and a function to return the
    appropriate reward strategy object.

"""
from enum import Enum

from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy
from envs.strategy.state_calc.basic_state_strategy import BasicStateStrategy
from envs.strategy.state_calc.bigger_state_normalized_strategy import BiggerStateNormalizedStrategy
from envs.strategy.state_calc.bigger_state_strategy import BiggerStateStrategy
from envs.strategy.state_calc.minimap_state_normalized_strategy import \
    MinimapStateNormalizedStrategy
from envs.strategy.state_calc.minimap_state_strategy import MinimapStateStrategy


class StateStrategyEnum(Enum):
    """
    Enumeration of available reward strategies.
    """
    BASIC_STATE_STRATEGY = 0
    BIGGER_STATE_STRATEGY = 1
    BIGGER_STATE_STRATEGY_NORMALIZED = 2
    MINIMAP_STATE_STRATEGY = 3
    MINIMAP_STATE_NORMALIZED_STRATEGY = 4

    def return_strategy(self) -> AStateCalculationStrategy:
        """
        Returns an AStateCalculationStrategy object based on the selected reward strategy
         enumeration.

        Args:
            self: The selected state calculation strategy enumeration.

        Returns:
            An AStateCalculationStrategy object corresponding to the state calculation strategy
             enumeration returns BasicStateStrategy by default

        """
        # A dictionary that maps each enumeration to its corresponding reward strategy object.
        # It is slightly performance inefficient, because it should be class/instance level variable
        # but withing Enum class it is not possible to do.
        strategy_dict: dict[StateStrategyEnum, AStateCalculationStrategy] = {
            self.BASIC_STATE_STRATEGY: BasicStateStrategy(),
            self.BIGGER_STATE_STRATEGY: BiggerStateStrategy(),
            self.BIGGER_STATE_STRATEGY_NORMALIZED: BiggerStateNormalizedStrategy(),
            self.MINIMAP_STATE_STRATEGY: MinimapStateStrategy(),
            self.MINIMAP_STATE_NORMALIZED_STRATEGY: MinimapStateNormalizedStrategy()
        }

        try:
            return strategy_dict[self]
        except KeyError:
            return BasicStateStrategy()

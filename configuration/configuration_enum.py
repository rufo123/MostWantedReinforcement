"""
Module: This module defines an enumeration for different experiment configurations.

Classes:
- ConfigurationEnum: Enumeration for different experiment configurations.
"""
from enum import Enum

from configuration.experiments.fifth_experiment_removed_progress_reward import \
    FifthExperimentRemovedProgressReward
from configuration.experiments.first_experiment_small_state import FirstExperimentSmallState
from configuration.experiments.fourth_experiment_lap_terminal_scaled_reward import \
    FourthExperimentLapTerminalScaledReward
from configuration.experiments.second_experiment_bigger_state import SecondExperimentBiggerState
from configuration.experiments.sixth_experiment_minimap import SixthExperimentMinimap
from configuration.experiments.third_experiment_bigger_state_normalized import \
    ThirdExperimentBiggerStateNormalized
from configuration.i_configuration import IConfiguration


class ConfigurationEnum(Enum):
    """
    Enumeration for different experiment configurations.
    """

    FIRST_EXPERIMENT = 0
    SECOND_EXPERIMENT = 1
    THIRD_EXPERIMENT = 2
    FOURTH_EXPERIMENT = 3
    FIFTH_EXPERIMENT = 4
    SIXTH_EXPERIMENT = 5

    def return_configuration(self) -> IConfiguration:
        """
        Return an instance of a configuration object for the selected experiment.

        Returns:
            IConfiguration: An instance of a configuration object for the selected experiment.
        """

        # A dictionary that maps each enumeration to its corresponding reward strategy object.
        # It is slightly performance inefficient, because it should be class/instance level variable
        # but withing Enum class it is not possible to do.
        experiment_dict: dict[ConfigurationEnum, IConfiguration] = {
            self.FIRST_EXPERIMENT: FirstExperimentSmallState(),
            self.SECOND_EXPERIMENT: SecondExperimentBiggerState(),
            self.THIRD_EXPERIMENT: ThirdExperimentBiggerStateNormalized(),
            self.FOURTH_EXPERIMENT: FourthExperimentLapTerminalScaledReward(),
            self.FIFTH_EXPERIMENT: FifthExperimentRemovedProgressReward(),
            self.SIXTH_EXPERIMENT: SixthExperimentMinimap()
        }

        try:
            return experiment_dict[self]
        except KeyError:
            return FirstExperimentSmallState()

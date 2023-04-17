"""
Module: first_reward_strategy

This module contains the FirstRewardStrategy class which is an implementation of
 the ARewardStrategy abstract class.

Classes:
    FirstRewardStrategy

"""
from car_states.car_state_in_environment import CarStateInEnvironment
from envs.strategy.reward.a_reward_strategy import ARewardStrategy
from utils.print_utils.printer import Printer


# pylint: disable=too-few-public-methods
# pylint: disable=R0801
class FourthMinimap40PercentRewardStrategy(ARewardStrategy):
    """
    This class is an implementation of the ARewardStrategy abstract class.
    This implementation gives:
     - positive reward for:
        Completing The Race (Partially - 40%)

     - negative reward for:
        Not Completing The Race in specified count_of_steps
        
    """

    # noinspection DuplicatedCode
    def evaluate_reward(self, par_env_inputs: CarStateInEnvironment,
                        par_game_steps_per_episode: int,
                        par_env_steps_counter: int,
                        par_terminal: bool) -> tuple[float, bool]:
        """
        This method calculates the reward of the current step for the ShortRaceEnv environment.

        Args:
            par_env_inputs (CarStateInEnvironment): Object containing car state represented by 
                the environment
            par_game_steps_per_episode (int): Count of Configured Game Steps per Env Episode
            par_env_steps_counter: (int) Count of passed game Steps in Env
            par_terminal (bool): If the environment has reached a terminal state.

        Returns:
            Tuple[float, bool]: The reward value and if the episode is finished.
        """
        reward: float = 0
        terminal: bool = par_terminal

        # Ako daleko som od idealnej linie?

        # Fiat Punto Top Speed - 179 # Zatial docasne prec

        # 0 - 50 - Negative Reward ((-1) - 0)
        # if -1 >= tmp_speed < 50:
        # reward += (((50 - tmp_speed) / 50) / 255) * -1
        # 50 - 100 - Positive Reward ( 0 - 1)
        # elif 50 <= tmp_speed <= 100:
        # reward += (((tmp_speed - 50) / 50) / 255)
        # 100 - 179 - Reward 1 - (-1)
        # else:
        # reward += (((179 - tmp_speed) / 39.5) - 1) / 255

        tmp_normalization_value: int = par_game_steps_per_episode

        reward += self.__lap_progress_reward(par_env_inputs.lap_progress_difference,
                                             tmp_normalization_value)

        if par_env_steps_counter >= par_game_steps_per_episode or par_env_inputs.lap_progress >= 40:
            terminal = True
            if par_env_steps_counter >= par_game_steps_per_episode:
                Printer.print_info("Exceeded Step Limit", "FOURTH_REWARD_STRATEGY", )
                reward += ((par_env_inputs.lap_progress / 20) - 1)
            if par_env_inputs.lap_progress >= 40:
                reward += 1
                Printer.print_success("Lap Complete", "FOURTH_REWARD_STRATEGY")
            Printer.print_info("TERMINAL STATE ACHIEVED", "FOURTH_REWARD_STRATEGY")
        return reward, terminal

    def __lap_progress_reward(self, par_lap_progress_diff: float,
                              par_normalization_value: int) -> float:
        """
        Calculates the lap progress reward based on the difference in lap progress
        between the current and previous time step.

        :param par_lap_progress_diff: A float representing the difference in lap
            progress between the current and previous time step. The value should
            be between -1 and 1, where negative values represent falling behind
            and positive values represent making progress.
         :param par_normalization_value: An integer representing the normalization
            value to use in the reward calculation. This value should be greater
            than zero to avoid division by zero errors.

        :return: A float representing the lap progress reward. The value will be
            positive if the agent is making progress and negative if the agent
            is falling behind. The magnitude of the reward will be proportional
            to the magnitude of the lap progress difference, divided by the
            normalization value.
        """
        Printer.print_basic("Progress: " + str(par_lap_progress_diff), "FOURTH_REWARD_STRATEGY")
        return par_lap_progress_diff / par_normalization_value

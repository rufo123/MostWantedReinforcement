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
class FirstRewardStrategy(ARewardStrategy):
    """
    This class is an implementation of the ARewardStrategy abstract class.
    This implementation gives:
     - positive reward for:
        Offset (Distance From Road Centre): <0, 1> and <-1, 0>
        
        Completing The Race (Partially - 10%)
        
     - negative reward for:
        Offset (Distance From Road Centre): (-10, -1> and (-inf, -10> 
        
        Not Completing The Race in specified count_of_steps
    """

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

        reward += self.__distance_offset_reward(par_env_inputs.distance_offset_center,
                                                tmp_normalization_value)

        if par_env_steps_counter >= par_game_steps_per_episode or par_env_inputs.lap_progress >= 10:
            terminal = True
            if par_env_steps_counter >= par_game_steps_per_episode:
                Printer.print_info("Exceeded Step Limit", "FIRST_REWARD_STRATEGY", )
                reward += ((par_env_inputs.lap_progress / 5) - 1)
            if par_env_inputs.lap_progress >= 10:
                reward += 1
                Printer.print_success("Lap Complete", "FIRST_REWARD_STRATEGY")
                print()
            Printer.print_info("TERMINAL STATE ACHIEVED", "FIRST_REWARD_STRATEGY")
        return reward, terminal

    def __distance_offset_reward(self, par_car_distance_offset: float,
                                 par_normalization_value: int) -> float:
        """
        Computes the offset reward based on the car's distance offset from the target.

        Args:
            par_car_distance_offset (float): The car's distance offset from the target.
                Negative values mean the car is behind the target, positive values mean
                the car is ahead of the target.
            par_normalization_value (int): A normalization value used to scale the reward.

        Returns:
            float: The computed offset reward.

        The offset reward is computed as follows:
        - If the car's distance offset is between -1 and -10, a negative reward is given
          proportional to the offset's normalized value divided by the normalization value.
          The normalized offset is obtained by dividing the offset minus -1 by 9.
        - If the car's distance offset is lower than -10, a negative reward of -1 divided by
          the normalization value is given.
        - If the car's distance offset is between 0 and 1, a positive reward is given
          proportional to 1 divided by the normalization value.
        - If the car's distance offset is between -1 and 0, a positive reward is given
          proportional to (1 + offset) divided by the normalization value.
        """
        # Offset Reward

        offset_reward: float = 0

        if -1 > par_car_distance_offset >= -10:
            # Negative Reward - Offset Between - ( -10, -1 >
            tmp_normalized_offset_div_10: float = (par_car_distance_offset - (-1)) / 9
            offset_reward = tmp_normalized_offset_div_10 / par_normalization_value
        elif par_car_distance_offset < -10:
            # Negative Reward - Offset Greater Than 10 or Lower Than -10
            offset_reward = -1 / par_normalization_value
        elif par_car_distance_offset > 0:
            # Positive Reward - Offset <0, 1>
            offset_reward = 1 / par_normalization_value
        elif -1 <= par_car_distance_offset <= 0:
            # Positive Reward - Offset (-1, 0)
            offset_reward = (1 + par_car_distance_offset) / par_normalization_value

        return offset_reward

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
        Printer.print_basic("Progress: " + str(par_lap_progress_diff), "FIRST_REWARD_STRATEGY")
        return par_lap_progress_diff / par_normalization_value

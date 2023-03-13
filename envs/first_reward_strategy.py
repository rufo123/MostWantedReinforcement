"""
Module: first_reward_strategy

This module contains the FirstRewardStrategy class which is an implementation of
 the ARewardStrategy abstract class.

Classes:
    FirstRewardStrategy

"""
from envs.a_reward_strategy import ARewardStrategy


# pylint: disable=too-few-public-methods
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

    def evaluate_reward(self, par_env_inputs: tuple[float, float, float, float, float],
                        par_game_steps_per_episode: int,
                        par_env_steps_counter: int,
                        par_terminal: bool) -> tuple[float, bool]:
        """
        This method calculates the reward of the current step for the ShortRaceEnv environment.

        Args:
            par_env_inputs (tuple[float, float, float, float]): The current state of the
                environment.
            par_game_steps_per_episode (int): Count of Configured Game Steps per Env Episode
            par_env_steps_counter: (int) Count of passed game Steps in Env
            par_terminal (bool): If the environment has reached a terminal state.

        Returns:
            Tuple[float, bool]: The reward value and if the episode is finished.
        """
        reward: float = 0
        terminal: bool = par_terminal

        # tmp_speed: float = par_env_inputs[0]
        tmp_car_distance_offset: float = par_env_inputs[1]
        tmp_lap_progress: float = par_env_inputs[2]
        tmp_lap_progress_diff: float = par_env_inputs[3]
        # tmp_car_direction_offset = par_env_inputs[4]

        tmp_normalization_value: int = par_game_steps_per_episode

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

        # Lap Progress Percent Reward + upravit na presnejsie jednotky
        print("Progress: " + str(tmp_lap_progress_diff))

        # 255 -  nedelit 2 krat - len raz delit a poctom max stepov
        reward += (tmp_lap_progress_diff / tmp_normalization_value)

        # Offset Reward
        if -1 > tmp_car_distance_offset >= -10:
            # Negative Reward - Offset Between - ( -10, -1 >
            tmp_normalized_offset_div_10: float = (tmp_car_distance_offset - (-1)) / 9
            reward += tmp_normalized_offset_div_10 / tmp_normalization_value
        elif tmp_car_distance_offset < -10:
            # Negative Reward - Offset Greater Than 10 or Lower Than -10
            reward += -1 / tmp_normalization_value
        elif tmp_car_distance_offset >= 0:
            # Positive Reward - Offset <0, 1>
            reward += (1 / tmp_normalization_value)
        elif -1 <= tmp_car_distance_offset < 0:
            # Positive Reward - Offset <-1, 0>
            reward += ((1 + abs(tmp_car_distance_offset)) / tmp_normalization_value)

        if par_env_steps_counter >= par_game_steps_per_episode or tmp_lap_progress >= 10:
            terminal = True
            if par_env_steps_counter >= par_game_steps_per_episode:
                print("Exceeded Step Limit")
                reward += -1
            if tmp_lap_progress >= 10:
                reward += 1
                print("Lap Complete")
            print("Terminal")
        return reward, terminal

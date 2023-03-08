import multiprocessing
import time

import torch
from absl import flags

from action_translator_enum import ActionTranslatorEnum
from game_inputs import GameInputs
from utils.singleton.controls import Controls

FLAGS = flags.FLAGS
FLAGS([''])

a_lap_percent_curr: float

a_game_speed: float

a_reward: float


class Env:
    a_game_inputs: GameInputs
    a_step_counter: int

    default_settings = {
        'step_mul': 0,
        'game_steps_per_episode': 150,
        'visualize': True,
        'realtime': False
    }

    def __init__(self, par_game_inputs: GameInputs):
        super().__init__()
        self.a_game_speed = None
        self.a_reward = None
        self.env = None
        self.action_counter = 0
        self.controls = Controls()
        self.game_steps_per_episode: int = self.default_settings['game_steps_per_episode']

        self.a_game_inputs: GameInputs = par_game_inputs
        self.a_lap_percent_curr = 0.00

    def make_state(self):
        terminal = False
        tmp_reward = self.a_reward

        if self.a_game_inputs.agent_inputs_state is None:
            pass
        print("Debug Call")

        tmp_tuple_with_values: tuple = self.a_game_inputs.agent_inputs_state.get()

        tmp_speed: float = tmp_tuple_with_values[0]
        tmp_car_distance_offset: float = tmp_tuple_with_values[1]
        tmp_lap_progress: float = tmp_tuple_with_values[2]
        tmp_car_direction_offset: int = tmp_tuple_with_values[3]

        state = torch.tensor([tmp_speed, tmp_car_distance_offset,
                              tmp_lap_progress, tmp_car_direction_offset])

        state = state.numpy()

        return state, tmp_reward, terminal

    def reset(self):
        print("Debug Call")
        self.controls.release_all_keys()
        state, _, _ = self.make_state()

        tmp_queue_game_inputs: multiprocessing.Queue = self.a_game_inputs.game_initialization_inputs.get()
        self.a_game_speed = tmp_queue_game_inputs[1]
        self.a_game_inputs.game_initialization_inputs.put(tmp_queue_game_inputs)
        self.a_reward = 0

        self.controls.a_is_executing_critical_action = True
        self.a_game_inputs.game_restart_inputs.put(True)
        print("Restart Initiated")
        while not self.a_game_inputs.game_restart_inputs.empty():
            print("Waiting For Game To Restart")
            time.sleep(1)
        self.a_step_counter = 0
        self.controls.a_is_executing_critical_action = False
        return state

    def get_lap_progress_dif(self, par_lap_progress: float) -> float:
        return round(par_lap_progress - self.a_lap_percent_curr, 2)

    def update_lap_curr(self, par_new_lap_percent: float) -> None:
        self.a_lap_percent_curr = par_new_lap_percent

    def step(self, action):
        print("Step")
        terminal = False
        reward = self.a_reward
        self.take_action(action, 1 / self.a_game_speed)
        # daj off2set
        # daj progress - +1% - prida reward
        # daj progress - -1% - da pokutu

        tmp_tuple_with_values: tuple = self.a_game_inputs.agent_inputs_state.get()

        tmp_speed: float = tmp_tuple_with_values[0]
        tmp_car_distance_offset: float = tmp_tuple_with_values[1]
        tmp_lap_progress: float = tmp_tuple_with_values[2]
        tmp_car_direction_offset: float = tmp_tuple_with_values[3]

        # Ako daleko som od idealnej linie?

        # Fiat Punto Top Speed - 179 # Zatial docasne prec

        # 0 - 50 - Negative Reward ((-1) - 0)
        # if -1 >= tmp_speed < 50:
        # reward += (((50 - tmp_speed) / 50) / 255) * -1
        # 50 - 100 - Positive Reward ( 0 - 1)
        # elif 50 <= tmp_speed <= 100:
        # reward += (((tmp_speed - 50) / 50) / 255)
        # 100 - 179 - Reward 1 - (-1)
        # else:sd
        # reward += (((179 - tmp_speed) / 39.5) - 1) / 255

        # Lap Progress Percent Reward + upravit na presnejsie jednotky
        print("Progress: " + str(self.get_lap_progress_dif(tmp_lap_progress)))

        tmp_lap_progress_difference = self.get_lap_progress_dif(tmp_lap_progress)
        
        tmp_normalization_value: int = self.game_steps_per_episode

        # 255 -  nedelit 2 krat - len raz delit a poctom max stepov
        if tmp_lap_progress_difference > 0:
            reward += (tmp_lap_progress_difference / tmp_normalization_value)
        self.update_lap_curr(tmp_lap_progress)

        # Offset Reward
        if -1 > tmp_car_distance_offset >= -10:
            # Negative Reward - Offset Between - ( -10, -1 >
            tmp_normalized_offset_div_10: float = (tmp_car_distance_offset - (-1)) / 9
            reward += tmp_normalized_offset_div_10 / tmp_normalization_value
        elif tmp_car_distance_offset < -10:
            # Negative Reward - Offset Greater Than 10 or Lower Than -10
            reward += -1 / tmp_normalization_value
        #elif tmp_car_distance_offset >= 0:
            # Positive Reward - Offset <0, 1>
            #reward += (1 / tmp_normalization_value)
        #elif -1 <= tmp_car_distance_offset < 0:
            # Positive Reward - Offset <-1, 0>

        new_state = torch.tensor([tmp_speed, tmp_car_distance_offset,
                                  tmp_lap_progress, tmp_car_direction_offset])
        print("Rewardo: " + str(reward))
        self.a_reward = reward
        self.a_step_counter += 1
        if self.a_step_counter >= self.game_steps_per_episode or tmp_lap_progress >= 10:
            terminal = True
            if self.a_step_counter >= self.game_steps_per_episode:
                print("Exceeded Step Limit")
                self.a_reward += -1
            if tmp_lap_progress >= 10:
                self.a_reward += 1
                print("Lap Complete")
            print("Terminal")

        return new_state, reward, terminal

    def take_action(self, par_action: int, par_sleep_time: float = 1) -> int:
        return ActionTranslatorEnum(par_action).take_action(self.controls, par_sleep_time)

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()


def create_env(par_game_inputs: GameInputs) -> Env:
    return Env(par_game_inputs)

import multiprocessing
import time

import numpy as np
import torch

from absl import flags

from Utils.controls import Controls
from Game import Game

FLAGS = flags.FLAGS
FLAGS([''])

a_lap_percent_curr: float

a_game_speed: float

a_reward: float


class Env:
    a_queue_agent_inputs: multiprocessing.Queue
    a_queue_game_vars: multiprocessing.Queue
    a_queue_restart_game_input: multiprocessing.Queue
    a_step_counter: int

    default_settings = {
        'step_mul': 0,
        'game_steps_per_episode': 0,
        'visualize': True,
        'realtime': False
    }

    def __init__(self, par_queue_env_inputs: multiprocessing.Queue,
                 par_queue_game_vars: multiprocessing.Queue,
                 par_queue_restart_game_input: multiprocessing.Queue):
        super().__init__()
        self.a_game_speed = None
        self.a_reward = None
        self.env = None
        self.action_counter = 0
        self.controls = Controls()

        self.a_queue_agent_inputs = par_queue_env_inputs
        self.a_queue_game_vars = par_queue_game_vars
        self.a_queue_restart_game_input = par_queue_restart_game_input
        self.a_lap_percent_curr = 0.00

    def make_state(self):
        terminal = False
        tmp_reward = self.a_reward

        if self.a_queue_agent_inputs is None:
            x = 10
        print("Debug Call")

        tmp_tuple_with_values: tuple = self.a_queue_agent_inputs.get()

        tmp_speed: float = tmp_tuple_with_values[0]
        tmp_car_offset: float = tmp_tuple_with_values[1]
        tmp_lap_progress: float = tmp_tuple_with_values[2]

        state = torch.tensor([tmp_speed, tmp_car_offset,
                              tmp_lap_progress])

        state = state.numpy()

        return state, tmp_reward, terminal

    def reset(self):
        print("Debug Call")
        self.controls.ReleaseAllKeys()
        state, _, _ = self.make_state()

        tmp_queue_game_inputs: multiprocessing.Queue = self.a_queue_game_vars.get()
        self.a_game_speed = tmp_queue_game_inputs[1]
        self.a_queue_game_vars.put(tmp_queue_game_inputs)
        self.a_reward = 0

        self.a_queue_restart_game_input.put(True)
        print("Restart Initiated")
        while not self.a_queue_restart_game_input.empty():
            print("Waiting For Game To Restart")
            time.sleep(1)
        self.a_step_counter = 0
        return state

    def get_lap_progress_dif(self, par_lap_progress: float) -> float:
        return par_lap_progress - self.a_lap_percent_curr

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

        tmp_tuple_with_values: tuple = self.a_queue_agent_inputs.get()

        tmp_speed: float = tmp_tuple_with_values[0]
        tmp_car_offset: float = tmp_tuple_with_values[1]
        tmp_lap_progress: float = tmp_tuple_with_values[2]


        # Ako daleko som od idealnej linie?


        # Fiat Punto Top Speed - 179 # Zatial docasne prec

        # 0 - 50 - Negative Reward ((-1) - 0)
        if -1 >= tmp_speed < 50:
            reward += (((50 - tmp_speed) / 50) / 255) * -1
        # 50 - 100 - Positive Reward ( 0 - 1)
        elif 50 <= tmp_speed <= 100:
            reward += (((tmp_speed - 50) / 50) / 255)
        # 100 - 179 - Reward 1 - (-1)
        else:
            reward += (((179 - tmp_speed) / 39.5) - 1) / 255

        # Lap Progress Percent Reward + upravit na presnejsie jednotky
        if self.get_lap_progress_dif(tmp_lap_progress) == 1:
            reward += 1 / 255
            self.update_lap_curr(tmp_lap_progress)
        elif self.get_lap_progress_dif(tmp_lap_progress) == -1:
            reward += -1 / 255
            self.update_lap_curr(tmp_lap_progress)

        # Offset Reward
        if tmp_car_offset > 1:
            # Negative Reward
            reward += ((tmp_car_offset - 1) / 255) * -1
        elif tmp_car_offset < -1:
            # Negative Reward
            reward += ((tmp_car_offset - (-1)) / 255)
        elif 1 >= tmp_car_offset > 0:
            # Positive Reward - Offset 1 - 0
            reward += ((1 - abs(tmp_car_offset)) / 255)
        elif -1 <= tmp_car_offset < 0:
            # Positive Reward - Offset -1 - 0
            reward += ((1 - abs(tmp_car_offset)) / 255)
        else:
            # Positive Reward - Offset Equals 0
            reward += ((1 - abs(tmp_car_offset)) / 255)

        new_state = torch.tensor([tmp_speed, tmp_car_offset,
                                  tmp_lap_progress])
        print("Rewardo: " + str(reward))
        self.a_reward = reward
        self.a_step_counter += 1

        if self.a_step_counter >= 10:
            terminal = True
            print("Terminal")

        return new_state, reward, terminal

    def take_action(self, action, par_sleep_time: float = 1) -> int:
        print("Akcia")
        if action == 0:
            self.controls.Forward(par_sleep_time)
        elif action == 1:
            self.controls.ForwardRight(par_sleep_time)
        elif action == 2:
            self.controls.Right(par_sleep_time)
        elif action == 3:
            self.controls.BackwardRight(par_sleep_time)
        elif action == 4:
            self.controls.Backward(par_sleep_time)
        elif action == 5:
            self.controls.BackwardLeft(par_sleep_time)
        elif action == 6:
            self.controls.Left(par_sleep_time)
        else:
            self.controls.ForwardLeft()
        return action

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()


def create_env(par_queue_env_inputs: multiprocessing.Queue,
               par_queue_game_vars: multiprocessing.Queue,
               par_queue_restart_game_input: multiprocessing.Queue) -> Env:
    return Env(par_queue_env_inputs, par_queue_game_vars, par_queue_restart_game_input)

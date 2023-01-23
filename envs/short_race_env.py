import numpy as np
import torch

from absl import flags

from Utils.controls import Controls
from Game import Game

FLAGS = flags.FLAGS
FLAGS([''])

a_lap_percent_curr: float


class Env:
    default_settings = {
        'step_mul': 0,
        'game_steps_per_episode': 0,
        'visualize': True,
        'realtime': False
    }

    def __init__(self, par_game: Game):
        super().__init__()
        self.env = None
        self.action_counter = 0
        self.controls = Controls()
        self.game = par_game

        self.a_lap_percent_curr = 0.00

    def make_state(self):
        terminal = False
        reward = 0

        if self.game is None:
            x = 10

        state = torch.tensor([self.game.get_speed_mph(), self.game.get_car_offset(),
                              self.game.a_lap_progress.return_lap_completed_percent()])

        return state, reward, terminal

    def reset(self):
        state, _, _ = self.make_state()
        return state

    def get_lap_progress_dif(self) -> int:
        return self.game.a_lap_progress.return_lap_completed_percent() - self.a_lap_percent_curr

    def update_lap_curr(self, par_new_lap_percent: float) -> None:
        self.a_lap_percent_curr = self.game.a_lap_progress.return_lap_completed_percent()

    def step(self, action):
        terminal = False
        reward = 0
        self.take_action(action)
        # daj offset
        # daj progress - +1% - prida reward
        # daj progress - -1% - da pokutu

        # Lap Progress Percent Reward
        if self.get_lap_progress_dif() == 1:
            reward += 1
            self.update_lap_curr(self.game.a_lap_progress.return_lap_completed_percent())
        elif self.get_lap_progress_dif() == -1:
            reward += -1
            self.update_lap_curr(self.game.a_lap_progress.return_lap_completed_percent())

        # Offset Reward
        if self.game.get_car_offset() > 1 or self.game.get_game_speed() < -1:
            reward += -1
        else:
            reward += 1
        new_state = torch.tensor([self.game.get_speed_mph(), self.game.get_car_offset(),
                                  self.game.a_lap_progress.return_lap_completed_percent()])

        return new_state, reward, terminal

    def take_action(self, action):
        return {
            0: self.controls.Forward(),
            1: self.controls.ForwardRight(),
            2: self.controls.Right(),
            3: self.controls.BackwardRight(),
            4: self.controls.Backward(),
            5: self.controls.BackwardLeft(),
            6: self.controls.Left(),
        }.get(action, self.controls.ForwardLeft())

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()


def create_env(par_game: Game):
    return Env(par_game)

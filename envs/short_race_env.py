"""
The `short_race_env` module provides an environment for playing a game using Reinforcement Learning
"""
import multiprocessing
import time

import numpy as np
import torch
from absl import flags

from action_translator_enum import ActionTranslatorEnum
from car_states.car_state import CarState
from car_states.car_state_in_environment import CarStateInEnvironment
from envs.a_reward_strategy import ARewardStrategy
from envs.reward_strategy_enum import RewardStrategyEnum
from game_inputs import GameInputs
from utils.print_utils.printer import Printer
from utils.singleton.controls import Controls

FLAGS = flags.FLAGS
FLAGS([''])

a_lap_percent_curr: float

a_game_speed: float

a_reward: float


# pylint: disable=too-many-instance-attributes
class Env:
    """
    Environment class for RL training.
    """
    a_game_inputs: GameInputs
    a_step_counter: int
    a_reward_strategy: ARewardStrategy
    a_state_matrix: np.ndarray

    default_settings = {
        'step_mul': 0,
        'game_steps_per_episode': 150,
        'visualize': True,
        'realtime': False,
        'reward_strategy': RewardStrategyEnum.FIRST_REWARD_STRATEGY
    }

    def __init__(self, par_game_inputs: GameInputs):
        """
        Initializes an instance of the Env class.

        Args:
        par_game_inputs: A GameInputs object representing the input to the game.
        """
        super().__init__()
        self.a_game_speed: int = 1
        self.env = None
        self.action_counter = 0
        self.controls = Controls()
        self.game_steps_per_episode: int = self.default_settings['game_steps_per_episode']
        self.a_reward_strategy = self.default_settings['reward_strategy'].return_strategy()

        self.a_game_inputs: GameInputs = par_game_inputs
        self.a_lap_percent_curr = 0.00

    def make_state(self):
        """
        Generates the state tuple to be used in the next step of the environment.

        Returns:
        A tuple representing the state of the environment.
        """
        terminal = False
        tmp_reward: float = 0
        if self.a_game_inputs.agent_inputs_state is None:
            pass

        while self.a_game_inputs.agent_inputs_state.qsize() == 0:
            Printer.print_info("Waiting for Game API to send data (Make_State)", "ENV")
            time.sleep(1 / (self.a_game_speed * 2))
        tmp_tuple_with_values: tuple = self.a_game_inputs.agent_inputs_state.get()

        tmp_car_state_from_game: CarState = CarState(
            par_speed_mph=tmp_tuple_with_values[0],
            par_distance_offset_center=tmp_tuple_with_values[1],
            par_lap_progress=tmp_tuple_with_values[2],
            par_incline_center=tmp_tuple_with_values[3]
        )

        state = torch.tensor([tmp_car_state_from_game.speed_mph,
                              tmp_car_state_from_game.distance_offset_center,
                              tmp_car_state_from_game.lap_progress,
                              tmp_car_state_from_game.incline_center])

        state = state.numpy()

        return state, tmp_reward, terminal

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
        A numpy array representing the initial state of the environment.
        """
        self.controls.release_all_keys()
        state, _, _ = self.make_state()

        tmp_queue_game_inputs: multiprocessing.Queue = \
            self.a_game_inputs.game_initialization_inputs.get()
        # noinspection PyUnresolvedReferences
        self.a_game_speed = tmp_queue_game_inputs[1]
        self.a_game_inputs.game_initialization_inputs.put(tmp_queue_game_inputs)

        self.controls.a_is_executing_critical_action = True
        self.a_game_inputs.game_restart_inputs.put(True)
        Printer.print_info("Restart Initiated", "ENV")
        while not self.a_game_inputs.game_restart_inputs.empty():
            Printer.print_info("Waiting For Game To Restart", "ENV")
            time.sleep(1)
        self.a_step_counter = 0
        self.controls.a_is_executing_critical_action = False
        self.controls.release_all_keys()
        return state

    def get_lap_progress_dif(self, par_lap_progress: float) -> float:
        """
        Returns the difference between the current lap progress and the new lap progress.

        Args:
        par_lap_progress: A float representing the current lap progress.

        Returns:
        A float representing the difference between the current lap progress and the new 
            lap progress.
        """
        return round(par_lap_progress - self.a_lap_percent_curr, 2)

    def update_lap_curr(self, par_new_lap_percent: float) -> None:
        """
        Updates the current lap progress.

        Args:
        par_new_lap_percent: A float representing the new lap progress.
        """
        self.a_lap_percent_curr = par_new_lap_percent

    def step(self, action):
        """
        Takes an action in the environment and returns the next state, reward, and done flag.

        Args:
        action: An integer representing the action to take.

        Returns:
        A tuple containing the next state, reward, done flag, and steps taken.
        """
        self.a_step_counter += 1
        Printer.print_basic("--------------------")
        Printer.print_info("Step Internal Counter: " + str(self.a_step_counter), "ENV")
        terminal = False
        reward: float = 0
        self.take_action(action, 1 / self.a_game_speed)
        # daj off2set
        # daj progress - +1% - prida reward
        # daj progress - -1% - da pokutu
        while self.a_game_inputs.agent_inputs_state.qsize() == 0:
            Printer.print_info("Waiting for Game API to send data (Step)", "ENV")
            time.sleep(1 / (self.a_game_speed * 2))
        tmp_tuple_with_values: tuple = self.a_game_inputs.agent_inputs_state.get()

        tmp_speed: float = tmp_tuple_with_values[0]
        tmp_car_distance_offset: float = tmp_tuple_with_values[1]
        tmp_lap_progress: float = tmp_tuple_with_values[2]
        tmp_car_direction_offset: float = tmp_tuple_with_values[3]

        tmp_lap_progress_diff: float = self.get_lap_progress_dif(tmp_lap_progress)

        edited_car_state: CarStateInEnvironment \
            = CarStateInEnvironment(par_speed_mph=tmp_speed,
                                    par_distance_offset_center=tmp_car_distance_offset,
                                    par_lap_progress=tmp_lap_progress,
                                    par_lap_progress_difference=tmp_lap_progress_diff,
                                    par_incline_center=tmp_car_direction_offset)

        new_reward, terminal = self.a_reward_strategy.evaluate_reward(edited_car_state,
                                                                      self.game_steps_per_episode,
                                                                      self.a_step_counter,
                                                                      terminal)
        reward += new_reward

        self.update_lap_curr(tmp_lap_progress)

        new_state = torch.tensor([tmp_speed, tmp_car_distance_offset,
                                  tmp_lap_progress, tmp_car_direction_offset])

        return new_state, reward, terminal, self.a_step_counter

    def take_action(self, par_action: int, par_sleep_time: float = 1) -> int:
        """
        Takes an action in the game environment.

        :param par_action: The action to take.
        :param par_sleep_time: The time to sleep after the action is taken.
        :return: The current score after the action is taken.
        """
        return ActionTranslatorEnum(par_action).take_action(self.controls, par_sleep_time)

    def return_steps_count(self) -> int:
        """
        Gets: The current count of steps
        :return: The current count of steps
        """
        return self.a_step_counter

    def return_game_steps_per_episode(self) -> int:
        """
        Gets: Configured count of game steps per episode
        :return: configured count of game steps per episode
        """
        return self.game_steps_per_episode

    def close(self):
        """
        Closes the game environment.
        """
        if self.env is not None:
            self.env.close()
        super().close()  # pylint: disable=no-member


def create_env(par_game_inputs: GameInputs) -> Env:
    """
    Creates a game environment for playing the game.

    :param par_game_inputs: The game inputs for the environment.
    :return: An instance of the game environment.
    """
    return Env(par_game_inputs)

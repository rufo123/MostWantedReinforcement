"""
The `short_race_env` module provides an environment for playing a game using Reinforcement Learning
"""
import multiprocessing
import time

import numpy as np
from absl import flags

from action_translator_enum import ActionTranslatorEnum
from car_states.car_state import CarState
from car_states.car_state_in_environment import CarStateInEnvironment
from car_states.enabled_game_api_values import EnabledGameApiValues
from configuration.i_configuration import IConfiguration
from envs.strategy.reward.a_reward_strategy import ARewardStrategy
from envs.strategy.state_calc.a_state_calc_strategy import AStateCalculationStrategy
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
    a_state_calculation_strategy: AStateCalculationStrategy
    a_enabled_game_api_values: EnabledGameApiValues
    a_state_matrix: np.ndarray
    a_configuration: IConfiguration

    a_edited_car_state: CarStateInEnvironment

    default_settings = {
        'step_mul': 0,
        'game_steps_per_episode': 150,
        'visualize': True,
        'realtime': False,
    }

    def __init__(self, par_game_inputs: GameInputs,
                 par_configuration: IConfiguration
                 ):
        """
        Initializes an instance of the Env class.

        Args:
        par_game_inputs: A GameInputs object representing the input to the game.
        """
        super().__init__()
        print("KEK")
        self.a_game_speed: int = 1
        self.env = None
        self.action_counter = 0
        self.controls = Controls()
        self.game_steps_per_episode: int = self.default_settings['game_steps_per_episode']
        self.a_reward_strategy = par_configuration.return_reward_strategy()
        self.a_state_calculation_strategy = par_configuration.return_state_calc_strategy()
        self.a_enabled_game_api_values = par_configuration.return_enabled_game_api_values()

        self.a_configuration = par_configuration

        self.a_game_inputs: GameInputs = par_game_inputs
        self.a_lap_percent_curr = 0.00

        self.a_edited_car_state = CarStateInEnvironment()
        self.a_edited_car_state.reset_car_state()

        self.a_state_matrix = np.zeros((5, 5), dtype=float) - 1

        par_game_inputs.agent_settings_to_game.put((
            self.default_settings['visualize'],
            self.default_settings['realtime']
        ))

        print("Values")

    def make_state(self):
        """
        Generates the state tuple to be used in the next step of the environment.

        Returns:
        A tuple representing the state of the environment.
        """
        terminal = False
        tmp_reward: float = 0
        self.a_lap_percent_curr = 0.00
        if self.a_game_inputs.agent_inputs_state is None:
            pass

        while self.a_game_inputs.agent_inputs_state.qsize() == 0:
            Printer.print_info("Waiting for Game API to send data (Make_State)", "ENV")
            time.sleep(1 / (self.a_game_speed * 2))
        tmp_car_state_from_game: CarState = self.a_game_inputs.agent_inputs_state.get()
        while not tmp_car_state_from_game.has_non_default_values():
            tmp_car_state_from_game: CarState = self.a_game_inputs.agent_inputs_state.get()

        state = self.a_state_calculation_strategy.calculate_state(
            par_action_taken=-1,
            par_car_state=tmp_car_state_from_game
        )

        state = state.numpy()

        return state, tmp_reward, terminal

    def reset(self, iteration_number: int):
        """
        Resets the environment to its initial state.

        Returns:
        A numpy array representing the initial state of the environment.
        """
        Printer.print_error(str(iteration_number))
        self.a_reward_strategy = \
            self.a_configuration.return_reward_strategy(iteration_number)
        self.a_state_calculation_strategy = \
            self.a_configuration.return_state_calc_strategy(iteration_number)

        self.a_lap_percent_curr = 0.00
        self.controls.release_all_keys()
        self.controls.reset_directional_controls()
        self.a_edited_car_state.reset_car_state()
        state, _, _ = self.make_state()

        tmp_queue_game_inputs: multiprocessing.Queue = \
            self.a_game_inputs.game_initialization_inputs.get()
        # noinspection PyUnresolvedReferences
        self.a_game_speed = tmp_queue_game_inputs[1] if not self.default_settings['realtime'] else 1
        self.a_game_inputs.game_initialization_inputs.put(tmp_queue_game_inputs)

        self.controls.a_is_executing_critical_action = True
        self.a_game_inputs.game_restart_inputs.put(True)
        Printer.print_info("Restart Initiated", "ENV")
        while not self.a_game_inputs.game_restart_inputs.empty():
            Printer.print_info("Waiting For Game To Restart", "ENV")
            time.sleep(1)
        self.a_step_counter = 0
        self.controls.a_is_executing_critical_action = False
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
        self.a_edited_car_state.reset_car_state()
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
        tmp_car_state_from_game: CarState = self.a_game_inputs.agent_inputs_state.get()
        while not tmp_car_state_from_game.has_non_default_values():
            tmp_car_state_from_game: CarState = self.a_game_inputs.agent_inputs_state.get()

        Printer.print_info("SPEED " + str(tmp_car_state_from_game.speed_mph), "ENV")

        tmp_lap_progress_diff: float = \
            self.get_lap_progress_dif(tmp_car_state_from_game.lap_progress)

        self.a_edited_car_state.assign_values(
            par_lap_progress_difference=tmp_lap_progress_diff,
            par_car_state=tmp_car_state_from_game
        )

        new_reward, terminal = self.a_reward_strategy.evaluate_reward(self.a_edited_car_state,
                                                                      self.game_steps_per_episode,
                                                                      self.a_step_counter,
                                                                      terminal)
        reward += new_reward

        self.update_lap_curr(tmp_car_state_from_game.lap_progress)

        new_state = self.a_state_calculation_strategy.calculate_state(
            par_action_taken=action,
            par_car_state=self.a_edited_car_state
        )

        # new_state = torch.tensor([tmp_speed, tmp_car_distance_offset,
        #                          tmp_lap_progress, tmp_car_direction_offset])

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


def create_env(par_game_inputs: GameInputs,
               par_configuration: IConfiguration) -> Env:
    """
    Creates a game environment for playing the game.

    :param par_game_inputs: The game inputs for the environment.
    :param par_configuration: The specified configuration.
    :return: An instance of the game environment.
    """
    return Env(
        par_game_inputs,
        par_configuration
    )

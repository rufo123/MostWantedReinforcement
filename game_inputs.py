"""
This module provides GameInputs class representing the inputs to a game
"""
import multiprocessing as mp


class GameInputs:
    """
    A class representing the inputs to a game.

    Attributes:
        __a_agent_inputs_state (mp.Queue): Inputs describing state of the game, such as car speed, 
            car distance offset,lap progress, and road centre incline.
        __a_game_initialization_inputs (mp.Queue): Inputs given by initializing game, such as game
            speed and initialization state.
        __a_game_restart_inputs (mp.Queue): Inputs given if restart is needed.
    """

    # Inputs Describing State of The Game
    # e.g. Car Speed, Car Distance Offset, Lap Progress, Road Centre Incline
    __a_agent_inputs_state: mp.Queue
    # Inputs Given By Initializing Game
    # e.g. Game Speed, Initialization State
    __a_game_initialization_inputs: mp.Queue
    # Inputs Given If Restart is Needed
    __a_game_restart_inputs: mp.Queue
    # Settings from Agent to Game
    __a_agent_settings_to_game: mp.Queue

    def __init__(self, par_agent_inputs_state: mp.Queue, par_game_initialization_inputs: mp.Queue,
                 par_game_restart_inputs: mp.Queue, par_agent_settings_to_game: mp.Queue):
        self.__a_agent_inputs_state = par_agent_inputs_state
        self.__a_game_initialization_inputs = par_game_initialization_inputs
        self.__a_game_restart_inputs = par_game_restart_inputs
        self.__a_agent_settings_to_game = par_agent_settings_to_game

    @property
    def agent_inputs_state(self) -> mp.Queue:
        """
        Gets the inputs describing state of the game.

        Returns:
            mp.Queue: The queue containing the inputs.
        """
        return self.__a_agent_inputs_state

    @property
    def game_initialization_inputs(self) -> mp.Queue:
        """
        Gets the inputs given by initializing game.

        Returns:
            mp.Queue: The queue containing the inputs.
        """
        return self.__a_game_initialization_inputs

    @property
    def game_restart_inputs(self) -> mp.Queue:
        """
        Gets the inputs given by initializing game.

        Returns:
            mp.Queue: The queue containing the inputs.
        """
        return self.__a_game_restart_inputs

    @property
    def agent_settings_to_game(self) -> mp.Queue:
        """
        Gets the settings given by agent to a game

        Returns:
            mp.Queue: The queue containing the settings.
        """
        return self.__a_agent_settings_to_game

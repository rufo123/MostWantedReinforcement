"""
Module that defines the GameStateStarting class, a concrete subclass of AGameState.
"""
from state.a_game_state import AGameState


# pylint: disable=too-few-public-methods
class GameStateStarting(AGameState):
    """
    Concrete subclass of AGameState that represents the game state when the game is starting.

    Methods:
        - return_state_text(self) -> str: Returns a string representing the current state
            of the game.
    """

    def return_state_text(self) -> str:
        """
        Returns a string representing the current state of the game when the game is starting.

        Returns:
            str: A string representing the current state of the game when the game is starting.
        """
        return "Starting"

    def return_color_representation(self) -> tuple[int, int, int]:
        """
        Returns the ORANGE color as a color representation of the state.

        Returns:
            tuple[int, int, int]: A tuple containing the ORANGE color values of the yellow
             (0, 175, 255) color state representation, represented as BGR.
        """
        return 0, 175, 255

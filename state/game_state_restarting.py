"""
Module that defines the GameStateRestarting class, a concrete subclass of AGameState.
"""

from state.a_game_state import AGameState


# pylint: disable=too-few-public-methods
class GameStateRestarting(AGameState):
    """
    Concrete subclass of AGameState that represents the game state when the game is restarting.

    Methods:
        - return_state_text(self) -> str: Returns a string representing the current state of
         the game.
    """

    def return_state_text(self) -> str:
        """
        Returns a string representing the current state of the game when the game is restarting.

        Returns:
            str: A string representing the current state of the game when the game is restarting.
        """
        return "Restarting"

    def return_color_representation(self) -> tuple[int, int, int]:
        """
        Returns the BLACK color as a color representation of the state.

        Returns:
            tuple[int, int, int]: A tuple containing the BLACK color values of the white
             (0, 0, 0) color state representation, represented as BGR.
        """
        return 0, 0, 0

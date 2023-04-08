"""
Module that defines the GameStateNotConnected class, a concrete subclass of AGameState.
"""

from state.a_game_state import AGameState


# pylint: disable=too-few-public-methods
class GameStateNotConnected(AGameState):
    """
    Concrete subclass of AGameState that represents the game state when the agent is not connected
     to Game Api.

    Methods:
        - return_state_text(self) -> str: Returns a string representing the current state of
         the game.
    """

    def return_state_text(self) -> str:
        """
        Returns a string representing the current state of the game when the agent is not connected
         to the Game Api.

        Returns:
            str: A string representing the current state of the game when the agent is not
             connected to the Game Api.
        """
        return "Not Connected"

    def return_color_representation(self) -> tuple[int, int, int]:
        """
        Returns the RED color as a color representation of the state.

        Returns:
            tuple[int, int, int]: A tuple containing the BGR color values of the yellow
             (0, 0, 255w) color state representation, represented as BGR.
        """
        return 0, 0, 255

"""
Module that defines the GameStateTraining class, a concrete subclass of AGameState.
"""
from state.a_game_state import AGameState


# pylint: disable=too-few-public-methods
class GameStateTraining(AGameState):
    """
    Concrete subclass of AGameState that represents the game state when the agent is training.

    Methods:
        - return_state_text(self) -> str: Returns a string representing the current state
         of the game.
    """

    def return_state_text(self) -> str:
        """
        Returns a string representing the current state of the game when the agent is training.

        Returns:
            str: A string representing the current state of the game when the agent is training.
        """
        return "Training"

    def return_color_representation(self) -> tuple[int, int, int]:
        """
        Returns the PURPLE color as a color representation of the state.

        Returns:
            tuple[int, int, int]: A tuple containing the PURPLE color values of the yellow
             (159, 43, 104) color state representation, represented as BGR.
        """
        return 159, 43, 104

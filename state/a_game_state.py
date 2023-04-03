"""
This module defines an abstract base class 'AGameState' that provides a method to return a 
    state text as a string.
"""
from abc import ABC


# pylint: disable=too-few-public-methods
class AGameState(ABC):
    """
    Abstract base class that defines the interface for implementing a game state in a game.

    Methods:
        - return_state_text(self) -> str: Abstract method that returns a string representing the
            current state of the game.
          This method must be implemented by any concrete subclass of AGameState.
    """

    def return_state_text(self) -> str:
        """
        Abstract method that returns a string representing the current state of the game.

        Returns:
            str: A string representing the current state of the game.
        """
        return ""

    def return_color_representation(self) -> tuple[int, int, int]:
        """
        Returns the color representation of the state.
        
        Defaults to PURPLE (159, 43, 104)

        Returns:
            tuple[int, int, int]: A tuple containing the BGR values of the color state
             representation.
        """
        return 159, 43, 104

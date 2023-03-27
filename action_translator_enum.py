""" 
This module provides an Enumeration class representing the actions that can be taken by the agent.
    The members of the enumeration are named after the direction of the action,
    and their values indicate the angle (in multiples of 45 degrees) between the
    car's current direction and the direction of the action.
"""
from enum import Enum

from utils.print_utils.printer import Printer
from utils.singleton.controls import Controls


class ActionTranslatorEnum(Enum):
    """
    Enumeration class representing the actions that can be taken by the agent.
    The members of the enumeration are named after the direction of the action,
    and their values indicate the angle (in multiples of 45 degrees) between the
    car's current direction and the direction of the action.
    """
    FORWARD = 0
    FORWARD_RIGHT = 1
    RIGHT = 2
    BACKWARD_RIGHT = 3
    BACKWARD = 4
    BACKWARD_LEFT = 5
    LEFT = 6
    FORWARD_LEFT = 7

    def __str__(self):
        """
        Returns a string representation of the action, including the action name
        and its corresponding value in parentheses.
        """
        return f"{self.name} ({self.value})"

    def take_action(self, par_controls: Controls,
                    par_sleep_time: float = 1) -> int:
        """
        Take an action based on the current value of the enum.

        Parameters:
          - par_controls: An instance of the `Controls` class used to control the game.
          - par_sleep_time: The amount of time to sleep after taking an action.

        Returns:
          - The integer value of the action taken.
          """
        executed_correctly: bool = False
        action = self.value
        par_controls.release_all_keys()
        if action == 0:
            executed_correctly = par_controls.forward(par_sleep_time)
        elif action == 1:
            executed_correctly = par_controls.forward_right(par_sleep_time)
        elif action == 2:
            executed_correctly = par_controls.right(par_sleep_time)
        elif action == 3:
            executed_correctly = par_controls.backward_right(par_sleep_time)
        elif action == 4:
            executed_correctly = par_controls.backward(par_sleep_time)
        elif action == 5:
            executed_correctly = par_controls.backward_left(par_sleep_time)
        elif action == 6:
            executed_correctly = par_controls.left(par_sleep_time)
        else:
            executed_correctly = par_controls.forward_left(par_sleep_time)
        Printer.print_success("Action: " + str(self), "ACT_TRANS")
        if not executed_correctly:
            raise ValueError("Control didn't execute correctly")
        return action

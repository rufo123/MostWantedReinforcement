"""
Module for creating models for a reinforcement learning agent to play Need for Speed
 Most Wanted (2005).
"""

from torch import nn

from models.abstract_factory.a_short_race_factory import AShortRaceFactory


class ShortRaceFactoryForward(AShortRaceFactory):
    """
    Factory for creating fully connected neural network models for a reinforcement learning
     agent to play Need for Speed  Most Wanted (2005).
     """
    a_count_of_action: int
    a_count_of_inputs: int

    def __init__(self, par_count_of_actions=8, par_count_of_inputs=25):
        """
        Initialize the factory with the given number of actions.

        Args:
            par_count_of_actions (int): The number of possible actions the agent can take.
            par_count_of_inputs (int): The number of inputs for the agent to work with.
        """
        self.a_count_of_action = par_count_of_actions
        self.a_count_of_inputs = par_count_of_inputs

    def create_features_model(self) -> nn.Module:
        """
        Create a fully connected neural network model that takes in a flattened image and outputs a
         feature vector.

        Returns:
            nn.Module: The features model.
        """

        return nn.Sequential(
            nn.Linear(self.a_count_of_inputs, 96),
            nn.ReLU(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    def create_policy_model(self) -> nn.Module:
        """
        Create a fully connected neural network model that takes in a feature vector and outputs
         an action.

        Returns:
            nn.Module: The policy model.
        """
        return nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.a_count_of_action)
        )

    def create_value_model(self) -> nn.Module:
        """
        Create a fully connected neural network model that takes in a feature vector and outputs
         a value estimate.

        Returns:
            nn.Module: The value model.
        """
        return nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

"""
Module for creating models for a reinforcement learning agent to play  Need for Speed
 Most Wanted (2005).
"""
from typing import Union

from torch import nn

from models.abstract_factory.a_short_race_factory import AShortRaceFactory


class ShortRaceFactoryConvo(AShortRaceFactory):
    """
    Factory for creating convolutional neural network models for a reinforcement learning agent 
    to play Need for Speed  Most Wanted (2005).
    """

    a_count_of_action: int
    a_count_of_inputs: Union[int, tuple]
    a_count_of_features: int

    def __init__(self, par_count_of_actions=8,
                 par_count_of_inputs: Union[int, tuple] = (4, 48, 48),
                 par_count_of_features=9216):
        """
        Initialize the factory with the given number of actions and features.

        Args:
            par_count_of_actions (int): The number of possible actions the agent can take.
            par_count_of_inputs (int): The number of inputs for the agent to work with.
            par_count_of_features (int): The number of features in the input image.
        """
        self.a_count_of_action = par_count_of_actions
        self.a_count_of_inputs = par_count_of_inputs
        self.a_count_of_features = par_count_of_features

    def create_features_model(self, ) -> nn.Module:
        """
        Create a convolutional neural network model that takes in an image and outputs a feature
         vector.

        Returns:
            nn.Module: The features model.
        """
        # count_of_features = out_channels * (input_width // 2 ** num_maxpools) * (
        #           input_height // 2 ** num_maxpools)
        # count_of_features = 64 * (48 // 2 ** 2) * (48 // 2 ** 2)
        # count_of_features = 64 * 12 * 12
        # count_of_features = 9216
        return nn.Sequential(
            nn.Conv2d(in_channels=self.a_count_of_inputs[0], out_channels=32, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            # Ak male rozlisenie tak pridat pooling
            nn.Linear(self.a_count_of_features, 512),
            nn.ReLU()
        )

    def create_policy_model(self) -> nn.Module:
        """
        Create a fully connected neural network model that takes in a feature vector and outputs
         an action.

        Returns:
            nn.Module: The policy model.
        """
        return nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.a_count_of_action)
        )

    def create_value_model(self) -> nn.Module:
        """
        Create a fully connected neural network model that takes in a feature vector and outputs a
         value estimate.

        Returns:
            nn.Module: The value model.
        """
        return nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

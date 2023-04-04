"""
Module containing the abstract factory class for creating models for a reinforcement learning agent
 to play Need for Speed Most Wanted (2005).
"""
from abc import ABC, abstractmethod

from torch import nn


class AShortRaceFactory(ABC):
    """
    Abstract factory for creating models for a reinforcement learning agent to play Need for Speed
     Most Wanted (2005).
    """

    @abstractmethod
    def create_features_model(self) -> nn.Module:
        """
        Abstract method that creates the feature extraction model for the agent.

        Returns:
            nn.Module: The feature extraction model.
        """

    @abstractmethod
    def create_policy_model(self) -> nn.Module:
        """
        Abstract method that creates the policy model for the agent.

        Returns:
            nn.Module: The policy model.
        """

    @abstractmethod
    def create_value_model(self) -> nn.Module:
        """
        Abstract method that creates the value model for the agent.

        Returns:
            nn.Module: The value model.
        """

"""
Module for the PolicyValueModel class used to create a reinforcement learning agent
to play Need for Speed Most Wanted (2005).
"""

from torch import nn

from models.abstract_factory.a_short_race_factory import AShortRaceFactory
from utils.init import init_orthogonal_head, init_orthogonal_features


# pylint:disable=too-few-public-methods
class PolicyValueModel(nn.Module):
    """
    Class for creating a reinforcement learning agent to play Need for Speed Most Wanted (2005).
    """

    def __init__(self, par_short_race_factory: AShortRaceFactory):
        """
        Initialize PolicyValueModel object.

        Args:
        - par_short_race_factory (AShortRaceFactory): an object that creates PyTorch models for the
        agent to use

        Returns:
        - None
        """
        # pylint: disable=unused-argument, disable=super-with-arguments
        super(PolicyValueModel, self).__init__()
        self.model_factory = par_short_race_factory
        self.features_model = self.model_factory.create_features_model()
        self.features_model.apply(init_orthogonal_features)
        self.policy_model = self.model_factory.create_policy_model()
        self.policy_model.apply(init_orthogonal_head)
        self.value_model = self.model_factory.create_value_model()
        self.value_model.apply(init_orthogonal_head)

    def forward(self, par_tensor_x):
        """
        Forward pass through the neural network.

        Args:
        - par_tensor_x (torch.Tensor): input tensor for the neural network

        Returns:
        - tuple(torch.Tensor, torch.Tensor): output tensor of policy head and value head
        """
        features = self.features_model(par_tensor_x)
        policy = self.policy_model(features)
        value = self.value_model(features)
        return policy, value

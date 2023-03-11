"""
This module contains a PyTorch neural network model for a policy-value function, with an 
    architecture consisting
of three feedforward neural networks for feature extraction, policy prediction, and value
estimation.

The main class in this module is PolicyValueModel, which inherits from nn.Module and implements 
    the model's 
forward method, taking a tensor input and returning two tensors as output: the policy 
    probabilities and the 
value estimate.

Additionally, the module includes helper functions for orthogonal weight initialization.
"""
from torch import nn

from utils.init import init_orthogonal_head, init_orthogonal_features

# pylint: disable=too-few-public-methods
class PolicyValueModel(nn.Module):
    """
    A neural network model that outputs policy and value predictions from input states.

    Args:
        count_of_actions (int): The number of actions the model can predict.
        init_features_model (optional): A custom initialization method for the feature model.
        init_policy_model (optional): A custom initialization method for the policy model.
        init_value_model (optional): A custom initialization method for the value model.
    """

    def __init__(self, count_of_actions=8, init_features_model=None, init_policy_model=None,
                 init_value_model=None):
        # pylint: disable=unused-argument, disable=super-with-arguments
        super(PolicyValueModel, self).__init__()

        # input 2x84x84
        self.features_model = nn.Sequential(
            nn.Linear(4, 128),  # feature size = 1936
            nn.ReLU(),
            nn.Linear(128, 128),  # feature size = 1936
            nn.ReLU(),
            nn.Linear(128, 128),  # feature size = 1936
            nn.ReLU(),
        )
        self.features_model.apply(init_orthogonal_features)

        self.policy_model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, count_of_actions)
        )
        self.policy_model.apply(init_orthogonal_head)

        self.value_model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_model.apply(init_orthogonal_head)

    def forward(self, par_x):
        """
        Forward pass of the neural network model.

        Args:
            par_x: Input state.

        Returns:
            tuple: Policy and value predictions for the input state.
        """
        par_x = self.features_model(par_x)
        return self.policy_model(par_x), self.value_model(par_x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.init import init_orthogonal_head, init_orthogonal_features



class PolicyValueModel(nn.Module):
    def __init__(self, count_of_actions = 8, init_features_model = None, init_policy_model = None, init_value_model = None):
        super(PolicyValueModel, self).__init__()

        #input 2x84x84
        self.features_model = nn.Sequential(
            nn.Linear(3, 128),  # feature size = 1936
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

    def forward(self, x):
        x = self.features_model(x)
        return self.policy_model(x), self.value_model(x)

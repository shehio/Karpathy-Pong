import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor:
    def __init__(self, policy_network: nn.Module, action_space: list):
        self.policy_network = policy_network
        self.action_space = action_space

    def act(self, state: torch.Tensor) -> (torch.Tensor, any, torch.distributions.distribution):
        action_probabilities = self.policy_network(state)
        distribution = Categorical(action_probabilities)
        sampled_action_index = distribution.sample()
        action = self.action_space[sampled_action_index.item()]

        return sampled_action_index, action, distribution

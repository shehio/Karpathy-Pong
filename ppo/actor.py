import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor:
    def __init__(self, action_space: list, old_policy_network: nn.Module = None):
        if old_policy_network:
            self.old_policy_network = old_policy_network
        else:
            self.old_policy_network = Actor.Policy()

        self.action_space = action_space

    def act(self, state: torch.Tensor) -> (torch.Tensor, any, torch.distributions.distribution):
        action_probabilities = self.old_policy_network(state)
        distribution = Categorical(action_probabilities)
        sampled_action_index = distribution.sample()
        action = self.action_space[sampled_action_index.item()]

        return sampled_action_index, action, distribution

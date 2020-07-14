import torch
import torch.nn as nn


class Critic:
    def __init__(self, value_network: nn.Module):
        self.value_network = value_network

    def evaluate(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_network(state)

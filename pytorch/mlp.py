import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim.rmsprop as rmsprop
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_count, hidden_layers=[200, 200], output_count=3,
                 learning_rate=0.005, decay_rate=0.99, drop_out_rate=0.3):
        super(MLP, self).__init__()

        layers_count = self.construct_layers_count(hidden_layers, input_count)
        self.layers = self.create_network(layers_count, output_count, drop_out_rate)

        # update buffers that add up gradients over a batch
        self.optimizer = rmsprop.RMSprop(learning_rate=learning_rate, decay_rate=decay_rate)
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()} # rmsprop memory

    @staticmethod
    def construct_layers_count(hidden_layers, input_count):
        layers_count = hidden_layers
        layers_count.insert(0, input_count)
        return layers_count

    @staticmethod
    def create_network(layers_count, output_count, drop_out_rate):
        layers = nn.ModuleList()
        for i in range(layers_count - 1):
            MLP.add_layer_to_network(layers, layers_count[i], layers_count[i + 1])
        layers.append(nn.Dropout(drop_out_rate))
        layers.append(layers_count[-1], output_count)
        return layers

    @staticmethod
    def add_layer_to_network(layers, input_layer_count, output_layer_count):
        current_layer = nn.Linear(input_layer_count, output_layer_count, bias=False)
        layers.append(current_layer)
        layers.append(nn.ReLU)

    def forward_pass(self, _input):
        for layer in self.layers:
            _input = layer(_input)
        return F.softmax(_input)

    def backward_pass(self, eph, epdlogp, epx):
        pass

    def train(self, cumulative_rewards):
        loss = torch.sum(-cumulative_rewards, -1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

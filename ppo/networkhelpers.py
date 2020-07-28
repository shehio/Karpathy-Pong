import torch
import torch.nn as nn
import torch.optim.rmsprop as rmsprop
import torch.nn.functional as F


class CommonHelpers:
    @staticmethod
    def construct_layers_count(hidden_layers, input_count):
        layers_count = hidden_layers
        layers_count.insert(0, input_count)
        return layers_count

    @staticmethod
    def create_network(layers_count, output_count, drop_out_rate, tanh=False):
        layers = nn.ModuleList()
        for i in range(len(layers_count) - 1):
            CommonHelpers.__add_layer_to_network(layers, layers_count[i], layers_count[i + 1], tanh)
        layers.append(nn.Dropout(drop_out_rate))
        layers.append(nn.Linear(layers_count[-1], output_count))
        return layers

    @staticmethod
    def __add_layer_to_network(layers, input_layer_count, output_layer_count, tanh=False):
        current_layer = nn.Linear(input_layer_count, output_layer_count, bias=False)
        layers.append(current_layer)
        if tanh:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.ReLU(inplace=True))


class ActorNetwork(nn.Module):
    def __init__(self, input_count, hidden_layers, output_count, learning_rate=0.002,
                 decay_rate=0.99, dropout_rate=0.0, tanh=False):
        super(ActorNetwork, self).__init__()

        self.input_count = input_count
        layers_count = CommonHelpers.construct_layers_count(hidden_layers, input_count)
        self.layers = CommonHelpers.create_network(layers_count, output_count, dropout_rate, tanh)
        self.optimizer = rmsprop.RMSprop(self.parameters(), lr=learning_rate, weight_decay=decay_rate)

    def forward(self, _input: torch.tensor):
        _output = _input.float()
        for layer in self.layers:
            _output = layer(_output)

        return F.softmax(_output)


class CriticNetwork(nn.Module):
    def __init__(self, input_count, hidden_layers, output_count,learning_rate=0.002,
                 decay_rate=0.99, dropout_rate=0.0, tanh=False):
        super(CriticNetwork, self).__init__()

        self.input_count = input_count
        layers_count = CommonHelpers.construct_layers_count(hidden_layers, input_count)
        self.layers = CommonHelpers.create_network(layers_count, output_count, dropout_rate, tanh)
        self.optimizer = rmsprop.RMSprop(self.parameters(), lr=learning_rate, weight_decay=decay_rate)

    def forward(self, _input):
        _output = _input
        for layer in self.layers:
            _output = layer(_output)
        return _output


class NetworkHelpers:

    @staticmethod
    def create_simple_actor_network(
            input_count: int,
            hidden_layers: list,
            output_count: int,
            dropout_rate: float = 0.0,
            tanh=False):
        """
        :return: A simple actor network with either ReLU or Tanh in the intermediate layers
        and softmax in the final layer
        """
        return ActorNetwork(
            input_count=input_count,
            hidden_layers=hidden_layers,
            output_count=output_count,
            dropout_rate=dropout_rate,
            tanh=tanh)

    @staticmethod
    def create_simple_critic_network(
            input_count: int,
            hidden_layers: list,
            output_count: int,
            dropout_rate: float = 0.0,
            tanh=False):
        """
        :return: A simple critic network with either ReLU or Tanh in the all layers
        """
        return CriticNetwork(
            input_count=input_count,
            hidden_layers=hidden_layers,
            output_count=output_count,
            dropout_rate=dropout_rate,
            tanh=tanh)

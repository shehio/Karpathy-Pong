import torch.nn as nn
import torch.optim.rmsprop as rmsprop
import torch.nn.functional as F


class NetworkHelpers:

    @staticmethod
    def create_simple_network(tanh=False):
        '''
        :return: A simple network with either ReLU or Tanh in the intermediate layers and softmax in the final layer
        '''

        return NetworkHelpers.Network(tanh=tanh)

    class Network(nn.Module):
        def __init__(self, input_count, hidden_layers=[200, 200], output_count=3,
                     learning_rate=0.005, decay_rate=0.99, drop_out_rate=0.5, tanh=False):
            super(NetworkHelpers.Network, self).__init__()

            self.input_count = input_count
            layers_count = NetworkHelpers.__construct_layers_count(hidden_layers, input_count)
            self.layers = NetworkHelpers.__create_network(layers_count, output_count, drop_out_rate, tanh)
            self.optimizer = rmsprop.RMSprop(self.parameters(), lr=learning_rate, weight_decay=decay_rate)

        def forward(self, _input):
            _input = _input.view(-1, self.input_count)
            for layer in self.layers:
                _input = layer(_input)
            return F.softmax(_input)

        @staticmethod
        def __construct_layers_count(hidden_layers, input_count):
            layers_count = hidden_layers
            layers_count.insert(0, input_count)
            return layers_count

        @staticmethod
        def __create_network(layers_count, output_count, drop_out_rate, tanh=False):
            layers = nn.ModuleList()
            for i in range(len(layers_count) - 1):
                NetworkHelpers.__add_layer_to_network(layers, layers_count[i], layers_count[i + 1], tanh)
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

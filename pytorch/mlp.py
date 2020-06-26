import os
import torch
import torch.nn as nn
import torch.optim.rmsprop as rmsprop
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_count, hidden_layers=[200, 200], output_count=3,
                 learning_rate=0.005, decay_rate=0.99, drop_out_rate=0.5):
        super(MLP, self).__init__()

        self.input_count = input_count
        layers_count = self.construct_layers_count(hidden_layers, input_count)
        self.layers = self.create_network(layers_count, output_count, drop_out_rate)
        self.optimizer = rmsprop.RMSprop(self.parameters(), lr=learning_rate, weight_decay=decay_rate)

    @staticmethod
    def construct_layers_count(hidden_layers, input_count):
        layers_count = hidden_layers
        layers_count.insert(0, input_count)
        return layers_count

    @staticmethod
    def create_network(layers_count, output_count, drop_out_rate):
        layers = nn.ModuleList()
        for i in range(len(layers_count) - 1):
            MLP.add_layer_to_network(layers, layers_count[i], layers_count[i + 1])
        layers.append(nn.Dropout(drop_out_rate))
        layers.append(nn.Linear(layers_count[-1], output_count))
        return layers

    @staticmethod
    def add_layer_to_network(layers, input_layer_count, output_layer_count):
        current_layer = nn.Linear(input_layer_count, output_layer_count, bias=False)
        layers.append(current_layer)
        layers.append(nn.ReLU(inplace=True))

    def forward(self, _input):
        _input = _input.view(-1, self.input_count)
        for layer in self.layers:
            _input = layer(_input)
        return F.softmax(_input, dim=1)

    def update_policy(self, discounted_rewards, log_probs, dev=torch.device("cpu")):
        cumulative_reward = - torch.cat(log_probs).to(dev) * discounted_rewards
        loss = torch.sum(cumulative_reward, -1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_network(self, episode, network_file):
        print(f'Saving the model at {episode}')
        torch.save({'episode': episode,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   network_file)

    def load_network_and_episode(self, network_file):
        if os.path.exists(network_file):
            checkpoint = torch.load(network_file)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            episode = checkpoint['episode']
            print(f"Loaded existing model continuing from epoch {episode}")
            return episode
        return 0

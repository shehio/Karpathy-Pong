from torch.distributions import Categorical
from torch.optim.rmsprop import RMSprop

from .networkhelpers import NetworkHelpers
from memory import Memory


class Critic:
    def __init__(self, learning_rate=0.0001, decay_rate=0.9):
        self.new_policy_network = NetworkHelpers.Policy(tanh=True)
        self.value_function_network = NetworkHelpers.Policy(tanh=True)
        self.policy_optimizer = RMSprop(self.parameters(), lr=learning_rate, weight_decay=decay_rate)
        self.value_function_optimizer = RMSprop(self.parameters(), lr=learning_rate, weight_decay=decay_rate)

    def evaluate(self, memory: Memory):
        action_probabilities = self.new_policy_network(memory.states)
        state_values = self.value_function_network(memory.states)

        distribution = Categorical(action_probabilities)
        log_probabilities = distribution.log_prob(memory.actions)

        return log_probabilities, state_values, distribution.entropy()

    def train_policy_and_value_networks(self, loss):
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.value_function_optimizer.zero_grad()
        loss.backward()
        self.value_function_optimizer.step()

    def __sample_and_save_action(self, state):
        action_probabilities = self.new_policy_network(state)
        distribution = Categorical(action_probabilities)
        action = distribution.sample()
        chosen_action = self.action_space[action.item()]
        self.memory.dlogps.append(distribution.log_prob(action))
        return chosen_action, distribution

    def __save_value(self, state):
        value = self.value_function_network(state)
        self.memory.state_values.append(value)

    def __save_entropy(self, distribution):
        entropy = distribution.entropy()
        self.memory.entropies.append(entropy)

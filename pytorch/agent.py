from memory import Memory
from torch.distributions import Categorical

from helpers import Helpers
from .mlp import MLP


class Agent:
    def __init__(self, learning_rate, decay_rate, gamma=0.99, batch_size=5, load_network=True, network_file='mlp.pt'):
        self.memory = Memory()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.network_file = network_file
        self.policy_network = MLP(input_count=6400, hidden_layers=[128, 128], output_count=3,
                                  learning_rate=learning_rate, decay_rate=decay_rate, drop_out_rate=0.5)

        self.policy_network.train()

        if load_network:
            self.episode = self.__load_policy_network_and_episode()

    def get_action(self, state):
        probabilities = self.policy_network(state)
        distribution = Categorical(probabilities)
        action = distribution.sample()
        self.memory.dlogps.append(distribution.log_prob(action))
        custom_action = [1, 2, 3]
        action = custom_action[action.item()]
        return action

    def reap_reward(self, reward):
        self.memory.rewards.append(reward)

    def make_episode_end_updates(self):
        self.episode = self.episode + 1
        self.__train_policy_network()
        self.__save_policy_network()

    def __train_policy_network(self):
        if self.episode % self.batch_size == 0:
            d_rewards = Helpers.discount_and_normalize_rewards(self.memory.rewards, self.gamma)
            loss = self.policy_network.update_policy(d_rewards, self.memory.dlogps)
            self.memory = Memory()

    def __save_policy_network(self):
        if self.episode % (self.batch_size * 5) == 0:
            self.policy_network.save_network(self.episode, self.network_file)

    def __load_policy_network_and_episode(self):
        return self.policy_network.load_network_and_episode(self.network_file)

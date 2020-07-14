from memory import Memory
from torch.distributions import Categorical

from helpers import Helpers
from .mlp import MLP as PyMLP


class Agent:
    def __init__(self, action_space: list, policy_network: PyMLP, gamma=0.99,
                 batch_size=5, load_network=True, network_file='mlp.pt', frame_difference_enabled=True):
        self.action_space = action_space
        self.gamma = gamma
        self.memory = Memory()

        self.policy_network = policy_network
        self.policy_network.train()
        self.batch_size = batch_size
        self.network_file = network_file
        self.frame_difference_enabled = frame_difference_enabled

        if load_network:
            self.episode = self.__load_policy_network_and_episode()

    def get_action(self, state):
        probabilities = self.policy_network(state, self.frame_difference_enabled)  # -> list [3]
        distribution = Categorical(probabilities)
        action = distribution.sample()  # -> [0.5, 0.3, 0.2]

        self.memory.dlogps.append(distribution.log_prob(action))
        action = self.action_space[action.item()]

        return action

    def reap_reward(self, reward):
        self.memory.actual_rewards.append(reward)

    def make_episode_end_updates(self):
        self.episode = self.episode + 1
        self.__train_policy_network()
        self.__save_policy_network()

    def __train_policy_network(self):
        if self.episode % self.batch_size == 0:
            discounted_rewards = Helpers.discount_and_normalize_rewards_for_pong(self.memory.actual_rewards, self.gamma)
            loss = self.policy_network.update_policy(discounted_rewards, self.memory.dlogps)
            self.memory = Memory()

    def __save_policy_network(self):
        if self.episode % (self.batch_size * 5) == 0:
            self.policy_network.save_network(self.episode, self.network_file)

    def __load_policy_network_and_episode(self):
        return self.policy_network.load_network_and_episode(self.network_file)

from helpers import Helpers
from .mlp import MLP
from memory import Memory
from torch.distributions import Categorical


class Agent:
    def __init__(self, learning_rate, decay_rate, gamma=0.99, batch_size=5, load_network=True, network_file='save.p'):
        self.memory = Memory()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_network = MLP(input_count=6400, hidden_layers=[200, 200], output_count=3,
                                  learning_rate=learning_rate, decay_rate=decay_rate, drop_out_rate=0.3)

        if load_network:
            self.policy_network.load_network(network_file)

    def get_action(self, state):
        probabilities = self.policy_network(state)
        distribution = Categorical(probabilities)
        action = distribution.sample()
        self.memory.dlogps.append(distribution.log_prob(action))
        return action.item()

    def reap_reward(self, reward):
        self.memory.rewards.append(reward)

    def make_episode_end_updates(self, episode_number):
        self.__train_policy_network(episode_number)
        self.__save_policy_network(episode_number)

    def __train_policy_network(self, episode_number):
        if episode_number % self.batch_size == 0:
            batch_discounted_rewards = Helpers.discount_and_normalize_rewards(self.memory.rewards, self.gamma)
            cumulative_rewards = self.memory.dlogps * batch_discounted_rewards
            self.policy_network.train(cumulative_rewards)
            memory = Memory()

    def __save_policy_network(self, episode_number):
        if episode_number % 100 == 0:
            self.policy_network.save_network()

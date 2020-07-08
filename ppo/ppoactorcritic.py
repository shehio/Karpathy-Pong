import copy
import itertools
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop

from .actor import Actor
from .critic import Critic
from helpers import Helpers
from memory import Memory


class PPOActorCritic:
    def __init__(self, actor: Actor, critic: Critic, action_space: list, episode_number=0, gamma: float = 0.99,
                 eta: float = 0.2, c1: float = 0.5, c2: float = 0.01, batch_size: int = 10,
                 learning_rate: float = 0.002, decay_rate: float = 0.90, epochs: int = 4):
        self.actor = actor
        self.critic = critic
        self.action_space = action_space
        self.episode_number = episode_number

        self.gamma = gamma
        self.eta = eta
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epochs = epochs

        self.new_policy_network = copy.deepcopy(self.actor.policy_network)
        self.optimizer = self.__get_optimizers()
        self.memory = Memory()

    def get_action(self, state: np.array):
        state = torch.from_numpy(state)
        self.memory.states.append(state)

        with torch.no_grad():
            sampled_action_index, action, distribution = self.actor.act(state)

        self.memory.actions.append(sampled_action_index)
        self.memory.dlogps.append(distribution.log_prob(sampled_action_index))

        return action

    def reap_reward(self, reward):
        self.memory.actual_rewards.append(reward)

    def has_finished(self, done):
        self.memory.episode_complete.append(done)

    def make_episode_updates(self):
        self.episode_number = self.episode_number + 1

        if self.episode_number % self.batch_size == 0:
            self.__evaluate_and_train_networks()
            self.__reset_actor_and_memory()

    def __get_optimizers(self):
        params = [self.new_policy_network.parameters(), self.critic.value_network.parameters()]
        optimizer = torch.optim.Adam(itertools.chain(*params), lr=self.learning_rate, betas=(0.9, 0.999))

        # It Model never converges using RMSProp for some reason, investigate!
        # optimizer = RMSprop(itertools.chain(*params), lr=self.learning_rate, weight_decay=self.decay_rate)
        return optimizer

    def __evaluate_and_train_networks(self):
        old_states = torch.stack(self.memory.states).detach()
        old_action_log_probabilities = torch.stack(self.memory.dlogps).flatten().detach()
        old_actions = torch.stack(self.memory.actions).detach()

        rewards = Helpers.discount_and_normalize_rewards_for_lunar_lander(
            self.memory.actual_rewards,
            self.memory.episode_complete,
            self.gamma)

        for _ in range(self.epochs):
            action_probabilities = self.new_policy_network(old_states)
            distributions = Categorical(action_probabilities)
            new_action_log_probabilities = distributions.log_prob(old_actions)

            state_values = self.critic.evaluate(old_states)
            advantages = rewards - state_values.detach()

            policy_ratio = torch.exp(new_action_log_probabilities - old_action_log_probabilities)
            clipped_ratio = torch.clamp(policy_ratio, 1 - self.eta, 1 + self.eta)

            surrogate1 = policy_ratio * advantages
            surrogate2 = clipped_ratio * advantages
            action_loss = torch.min(surrogate1, surrogate2).mean()

            value_loss = ((state_values - rewards) ** 2).mean()
            entropy_loss = distributions.entropy().mean()

            # loss is the negative of the gain in the paper: https://arxiv.org/abs/1707.06347
            total_loss = - action_loss + self.c1 * value_loss - self.c2 * entropy_loss

            self.optimizer.zero_grad()
            total_loss.mean().backward()
            self.optimizer.step()

    def __reset_actor_and_memory(self):
        self.actor = Actor(self.new_policy_network, self.action_space)
        self.memory = Memory()

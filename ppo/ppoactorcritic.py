import copy
import itertools
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop

from .actor import Actor
from .critic import Critic
from helpers import Helpers
from memory import Memory


class PPOActorCritic:
    def __init__(self, actor: Actor, critic: Critic, action_space: list, episode_number=0, gamma: float = 0.9,
                 eta: float = 0.2, c1: float = 0.5, c2: float = 0.1, batch_size: int = 5, learning_rate: float = 0.0001,
                 decay_rate: float = 0.90):
        self.actor = actor
        self.critic = critic
        self.action_space = action_space
        self.episode_number = episode_number

        self.gamma = gamma
        self.eta = eta
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size

        self.memory = Memory()
        self.optimizer = self.__get_optimizers(decay_rate, learning_rate)

    def get_action(self, state):
        self.memory.states.append(state)

        with torch.no_grad():
            sampled_action_index, action, distribution = self.actor.act(state)
        self.memory.actions.append(sampled_action_index)
        self.memory.dlogps.append(distribution.log_prob(sampled_action_index))

        return action

    def reap_reward(self, reward):
        self.memory.rewards.append(reward)

    def make_episode_updates(self):
        self.episode_number = self.episode_number + 1
        if self.episode_number % self.batch_size == 0:
            self.memory.rewards = Helpers.discount_and_normalize_rewards(self.memory.rewards, self.gamma)
            self.__evaluate_and_train_networks(self.memory, copy.deepcopy(self.actor.policy_network))
            self.__reset_actor_and_memory()

    def __get_optimizers(self, decay_rate, learning_rate):
        params = [self.actor.policy_network.parameters(), self.critic.value_network.parameters()]
        optimizer = RMSprop(itertools.chain(*params), lr=learning_rate, weight_decay=decay_rate)
        return optimizer

    def __evaluate_and_train_networks(self, memory, new_policy_network):
        for action, state, old_log_probability, reward in zip(memory.actions, memory.states, memory.dlogps,
                                                              memory.rewards):  # TODO: Make memory a ppo memory
            predicted_value = self.critic.evaluate(state)
            new_distribution = Categorical(new_policy_network(state))
            new_log_probability = new_distribution.log_prob(action)

            action_loss, value_loss = self.__get_action_and_value_loss(old_log_probability, new_log_probability,
                                                                       predicted_value, reward,
                                                                       new_distribution.entropy())
            self.__optimize_networks(action_loss, value_loss, reward, predicted_value)

    def __get_action_and_value_loss(self, old_action_log_probability, new_action_log_probability,
                                    predicted_value, actual_reward, distribution_entropy):
        advantage = torch.FloatTensor(actual_reward - predicted_value)
        surrogate1, surrogate2 = self.__get_ppo_surrogate_functions(old_action_log_probability,
                                                                    new_action_log_probability, advantage)
        surrogates_loss = torch.min(surrogate1, surrogate2).mean()
        value_loss1 = F.mse_loss(actual_reward, predicted_value[0][0])
        value_loss2 = F.mse_loss(actual_reward.clone().detach(), predicted_value[0][0].detach().clone())

        # loss is negative of the gain in the paper: https://arxiv.org/abs/1707.06347
        action_loss = - surrogates_loss + torch.mul(self.c1,  value_loss1) - torch.mul(self.c2, distribution_entropy)

        return action_loss, value_loss2

    def __get_ppo_surrogate_functions(self, old_action_log_probability, new_action_log_probability, advantage):
        policy_ratio = torch.exp(new_action_log_probability - old_action_log_probability)
        clipped_ratio = torch.clamp(policy_ratio, 1 - self.eta, 1 + self.eta)

        surrogate1 = policy_ratio * advantage
        surrogate2 = clipped_ratio * advantage
        return surrogate1, surrogate2

    def __optimize_networks(self, action_loss, value_loss, actual_reward, predicted_value):
        self.optimizer.zero_grad()
        action_loss.backward()
        self.optimizer.step()

    def __reset_actor_and_memory(self):
        self.actor = Actor(self.policy_network, self.action_space)
        self.memory = Memory()

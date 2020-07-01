import torch
import torch.nn as nn

from memory import Memory
from .actor import Actor
from .critic import Critic


class ActorCritic:
    """
    Only for PPO
    """
    def __init__(self, actor: Actor, critic: Critic, episode_number=0, gamma: float = 0.9, eta: float = 0.2,
                 c1: float = 0.5, c2: float = 0.1, batch_size: int = 5):
        self.actor = actor
        self.critic = critic
        self.episode_number = episode_number
        self.gamma = gamma
        self.eta = eta
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.memory = Memory()

    def reap_reward(self, reward):
        self.memory.rewards.append(reward)

    def act(self, state):
        self.memory.states.append(state)

        sampled_action_index, action, distribution = self.actor.act(state)
        self.memory.actions.append(action)
        self.memory.dlogps.append(distribution.log_prob(sampled_action_index))

        return action

    def make_episode_end_updates(self, episode_number):
        self.episode_number = self.episode_number + 1
        self.__evaluate(episode_number)

    def __evaluate(self):
        if self.episode % self.batch_size == 0:
            episode_dlogs, vf_values, distribution_entropy = self.critic.evaluate(self.memory, self.gamma)
            surrogate_functions = self.__get_ppo_surrogate_functions(distribution_entropy, episode_dlogs, vf_values)

            # loss is negative of the gain in the paper: https://arxiv.org/abs/1707.06347
            ppo_loss = - surrogate_functions[0] + surrogate_functions[1] - surrogate_functions[2]
            self.critic.train_policy_and_value_networks(ppo_loss)

            self.actor.old_policy_network = self.critic.new_policy_network
            self.memory = Memory()

    def __get_ppo_surrogate_functions(self, distribution_entropy, episode_dlogs, vf_values):
        policy_ratio = torch.exp(episode_dlogs - self.memory.dlogps)
        clipped_ratio = torch.clamp(policy_ratio, 1 - self.eta, 1 + self.eta)
        advantages = torch.FloatTensor(self.memory.rewards - vf_values)
        surrogate1 = torch.min(policy_ratio * advantages, clipped_ratio * advantages)
        surrogate2 = self.c1 * nn.MSELoss(vf_values - self.memory.rewards)
        surrogate3 = self.c2 * distribution_entropy
        return [surrogate1, surrogate2, surrogate3]


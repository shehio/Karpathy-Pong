import numpy as np
import torch


class Helpers:
    @staticmethod
    def sigmoid(number):
        return 1.0 / (1.0 + np.exp(-number))  # sigmoid "squashing" function to interval [0,1]

    @staticmethod
    def preprocess_frame(observation, device=torch.device("cpu")):
        observation = observation[35:195]  # crop
        observation = observation[::2, ::2, 0]  # down-sample by factor of 2
        observation[observation == 144] = 0  # erase background (background type 1)
        observation[observation == 109] = 0  # erase background (background type 2)
        observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
        return torch.tensor(observation).view(-1).numpy()

    @staticmethod
    def discount_and_normalize_rewards_for_pong(rewards, gamma, normalize=True, device=torch.device("cpu")):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        if normalize:
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

        return discounted_rewards

    @staticmethod  # @Todo: Collect the common parts into one method.
    def discount_and_normalize_rewards_for_lunar_lander(rewards, episode_completed, gamma, device=torch.device("cpu")):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if episode_completed[t]:
                running_add = 0
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + + 1e-5)

        return discounted_rewards

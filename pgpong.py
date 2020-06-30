import numpy as np
import gym
import time
import torch

from helpers import Helpers
from pytorch.agent import Agent as PyAgent
from pytorch.mlp import MLP as PyMLP

# hyper-parameters
hidden_layers_count = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 5e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

resume = True
render = True
sleep_for_rendering_in_seconds = 0.02
pixels_count = 80 * 80
frame_difference_enabled = True


def render_game():
    if render:
        env.render()
        time.sleep(sleep_for_rendering_in_seconds)


def get_frame_difference(observation, _previous_frame, device=torch.device('cpu')):
    # pre-process the observation, set input to network to be difference image
    processed_frame = Helpers.preprocess_frame(observation, device)
    if _previous_frame is not None:
        return processed_frame - _previous_frame, processed_frame
    else:
        return torch.tensor(np.zeros(pixels_count)).type(torch.FloatTensor).to(device), processed_frame


if __name__ == '__main__':
    env = gym.make("Pong-v0")
    current_frame = env.reset()
    previous_frame, running_reward = None, None  # To compute the difference frame
    reward_sum = 0
    action_space = [1, 2, 3]
    policy_network = PyMLP(input_count=6400, hidden_layers=[128, 128], output_count=len(action_space),
                           learning_rate=learning_rate, decay_rate=decay_rate, drop_out_rate=0.5)

    agent = PyAgent(action_space, policy_network)
    episode_number = agent.episode

    while True:
        render_game()
        if frame_difference_enabled:
            state, previous_frame = get_frame_difference(current_frame, previous_frame)
        else:
            state = Helpers.preprocess_frame(current_frame)
        action = agent.get_action(state)
        current_frame, reward, done, info = env.step(action)
        agent.reap_reward(reward)
        reward_sum += reward

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))

        if done:
            episode_number += 1
            agent.make_episode_end_updates()

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

            current_frame = env.reset()
            reward_sum = 0
            previous_frame = None

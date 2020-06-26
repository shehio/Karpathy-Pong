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

resume = True  # resume from previous checkpoint?
render = True
sleep_for_rendering_in_seconds = 0.02

pixels_count = 80 * 80  # input dimensionality: 80x80 grid


def render_game():
    if render:
        env.render()
        time.sleep(sleep_for_rendering_in_seconds)


def get_frame_difference(_observation, _previous_frame):
    # pre-process the observation, set input to network to be difference image
    current_frame = Helpers.preprocess_frame(_observation)
    state = current_frame - _previous_frame if _previous_frame is not None else torch.tensor(
        np.zeros(pixels_count)).type(torch.FloatTensor).to(torch.device('cpu'))
    return current_frame, current_frame


if __name__ == '__main__':
    env = gym.make("Pong-v0")
    observation = env.reset()
    previous_frame, running_reward = None, None  # used in computing the difference frame
    reward_sum = 0
    action_space = [1, 2, 3]
    policy_network = PyMLP(input_count=6400, hidden_layers=[128, 128], output_count=len(action_space),
                           learning_rate=learning_rate, decay_rate=decay_rate, drop_out_rate=0.5)

    agent = PyAgent(action_space, policy_network)
    episode_number = agent.episode

    while True:
        render_game()
        state = Helpers.preprocess_frame(observation,
                                         torch.device("cpu"))  # get_frame_difference(observation, previous_frame)
        action = agent.get_action(state)
        observation, reward, done, info = env.step(action)
        agent.reap_reward(reward)
        reward_sum += reward

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))

        if done:
            episode_number += 1
            agent.make_episode_end_updates()

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

            observation = env.reset()
            reward_sum = 0
            previous_frame = None

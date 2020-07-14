import gym
import time
import torch

from pytorch.agent import Agent as PyAgent
from pytorch.mlp import MLP as PyMLP

from ppo.actor import Actor
from ppo.critic import Critic
from ppo.networkhelpers import NetworkHelpers
from ppo.ppoactorcritic import PPOActorCritic as PPOAgent

# hyper-parameters
batch_size = 20  # every how many episodes to do a param update?
learning_rate = 5e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
update_threshold = 2000

resume = True
render = False
sleep_for_rendering_in_seconds = 0.02
pixels_count = 80 * 80
frame_difference_enabled = True
algorithm = 'ppo'
env_name = "LunarLander-v2"


def render_game():
    if render:
        env.render()
        time.sleep(sleep_for_rendering_in_seconds)


def create_actor_network(input_count: int, hidden_layers: list, output_count: int):
    return NetworkHelpers.create_simple_actor_network(input_count, hidden_layers, output_count, tanh=True)


def create_critic_network(input_count: int, hidden_layers: list, output_count: int):
    return NetworkHelpers.create_simple_critic_network(input_count, hidden_layers, output_count, tanh=True)


if __name__ == '__main__':
    env = gym.make(env_name)
    state_dimension = env.observation_space.shape[0]
    current_state = env.reset()
    episode_reward = 0
    batch_average_reward = 0
    timestep = 0
    action_space = [0, 1, 2, 3]

    if algorithm == 'vanilla':
        policy_network = PyMLP(input_count=state_dimension, hidden_layers=[128, 128], output_count=len(action_space),
                               learning_rate=learning_rate, decay_rate=decay_rate, drop_out_rate=0.5)
        agent = PyAgent(action_space, policy_network)
        episode_number = agent.episode

    elif algorithm == 'ppo':
        actor = Actor(create_actor_network(state_dimension, [64, 64], len(action_space)), action_space)
        critic = Critic(create_critic_network(state_dimension, [64], 1))
        episode_number = 0
        avg_length = 0
        agent = PPOAgent(actor, critic, action_space, episode_number, batch_size=batch_size)

    while True:
        render_game()
        action = agent.get_action(current_state)
        current_state, reward, done, info = env.step(action)

        agent.reap_reward(reward)
        agent.has_finished(done)

        timestep += 1
        avg_length += 1
        episode_reward += reward

        if timestep % update_threshold == 0:
            agent.make_episode_updates()
            timestep = 0

        if done:
            episode_number += 1
            batch_average_reward += episode_reward

            if episode_number % batch_size == 0:
                avg_length = int(avg_length / batch_size)
                batch_average_reward = int(batch_average_reward / batch_size)
                print(f'Episode {episode_number}\t avg length: {avg_length} \t reward: {batch_average_reward}')
                batch_average_reward = 0
                avg_length = 0

            current_state = env.reset()
            episode_reward = 0

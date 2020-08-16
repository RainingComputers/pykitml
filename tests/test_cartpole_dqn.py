from pykitml.testing import pktest_graph, pktest_nograph

import pytest

@pktest_graph
def test_cartpole():
    import random

    import numpy as np
    import gym
    import pykitml as pk

    # Wrapper class around the environment
    class Environment:
        def __init__(self):
            self._env = gym.make('CartPole-v0')

        def reset(self):
            return self._env.reset()

        def step(self, action):
            obs, reward, done, _ = self._env.step(action)

            # Reward function, from
            # https://github.com/keon/deep-q-learning/blob/master/ddqn.py
            x, x_dot, theta, theta_dot = obs
            r1 = (self._env.x_threshold - abs(x)) / self._env.x_threshold - 0.8
            r2 = (self._env.theta_threshold_radians - abs(theta)) / self._env.theta_threshold_radians - 0.5
            reward = r1 + r2

            return np.array(obs), reward, done

        def close(self):
            self._env.close()

        def render(self):
            self._env.render()

    env = Environment()

    # Create DQN agent and train it
    agent = pk.DQNAgent([4, 64, 64, 2])
    agent.set_save_freq(100, 'cartpole_agent')
    agent.train(env, 500, pk.Adam(0.001), render=True)

    # Plot reward graph
    agent.plot_performance()

if __name__ == '__main__':
    test_cartpole.__wrapped__()

from pykitml.testing import pktest_graph

@pktest_graph
def test_cartpole():
    import gymnasium as gym
    import pykitml as pk

    # Wrapper class around the environment
    class Environment:
        def __init__(self):
            self._env = gym.make('CartPole-v1', render_mode="human")

        def reset(self):
            return self._env.reset()[0]

        def step(self, action):
            obs, reward, done, _, _ = self._env.step(action)

            x, _, theta, _ = obs
            x_threshold = self._env.env.env.env.x_threshold
            theta_threshold_radians = self._env.env.env.env.theta_threshold_radians

            # Reward function, from
            # https://github.com/keon/deep-q-learning/blob/master/ddqn.py            
            r1 = (x_threshold - abs(x)) / x_threshold - 0.8
            r2 = (theta_threshold_radians - abs(theta)) / theta_threshold_radians - 0.5
            reward = r1 + r2

            return obs, reward, done

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

"""Class to create gym environments.
"""

import gym
import numpy as np

from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame


MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG"
]

class OneHotWrapper(gym.ObservationWrapper):
    """
    Transforms the discrete state space into continous.
    For state i will return one-hot vector with index i
    """
    def __init__(self, env):
        super(OneHotWrapper, self).__init__(env)
        self.num_states = env.observation_space.n
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.num_states))

    def _observation(self, observation):
        one_hot_obs = np.zeros(self.num_states)
        one_hot_obs[observation] = 1.0
        return one_hot_obs


class EnvironmentCreator(object):
    """Create environment.

    Parameters
    ----------
    game: (string) Gives the game to play
    """
    def __init__(self, game):
        self.game = game
        if game == 'pong':
            env = gym.make("PongNoFrameskip-v4")
            env = ScaledFloatFrame(wrap_dqn(env))
        elif game == 'frozen':
            from gym.envs.toy_text import FrozenLakeEnv
            env = FrozenLakeEnv(desc=MAP, is_slippery=False)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
            env = OneHotWrapper(env)
        elif game == 'cartpole':
            env = gym.make("CartPole-v0")
        else:
            raise NotImplementedError("Game not recognized. Try pong / frozen / cartpole.")
        self.env = env

    def get_env(self):
        """Return the created environment.
        """
        return self.env

    def callback(self, it, episode_rewards):
        """Returns True when the environment is solved
        """
        if self.game == 'pong':
            goal = 20.0
        elif self.game == 'frozen':
            goal = 0.99
        elif self.game == 'cartpole':
            goal = 199.0
        else:
            raise NotImplementedError("Game not recognized. Try pong / frozen / cartpole.")

        is_solved = it > 100 and sum(episode_rewards[-101:-1]) / 100 >= goal
        return is_solved

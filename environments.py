"""Class to create gym environments.
"""

import gym

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

class EnvironmentCreator(object):
    """Create environment.

    Parameters
    ----------
    game: (string) Gives the game to play
    """
    def __init__(self, game, record=False, outdir="records"):
        self.game = game
        if game == 'pong':
            env = gym.make("PongNoFrameskip-v4")
            env = ScaledFloatFrame(wrap_dqn(env))
        elif game == 'frozen':
            from gym.envs.toy_text import FrozenLakeEnv
            from wrappers import OneHotWrapper
            env = FrozenLakeEnv(desc=MAP, is_slippery=False)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
            env = OneHotWrapper(env)
        elif game == 'cartpole':
            env = gym.make("CartPole-v0")
        elif game == 'doom':
            from ppaquette_gym_doom.wrappers import SetPlayingMode, SetResolution, ToDiscrete
            from wrappers import NoNegativeRewardEnv, BufferedObsEnv
            env = gym.make('ppaquette/DoomMyWayHome-v0')
            modewrapper = SetPlayingMode('algo')
            obwrapper = SetResolution('160x120')
            acwrapper = ToDiscrete('minimal')
            env = modewrapper(obwrapper(acwrapper(env)))

            if record:
                env = gym.wrappers.Monitor(env, outdir, force=True)
            fshape = (42, 42)

            env.seed(None)
            env = NoNegativeRewardEnv(env)
            env = BufferedObsEnv(env, skip=1, shape=fshape)
        else:
            raise NotImplementedError("Game not recognized. Try pong / frozen / cartpole / doom.")

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
        elif self.game == 'doom':
            goal = 0.99
        else:
            raise NotImplementedError("Game not recognized. Try pong / frozen / cartpole.")

        is_solved = it > 100 and sum(episode_rewards[-101:-1]) / 100 >= goal
        return is_solved

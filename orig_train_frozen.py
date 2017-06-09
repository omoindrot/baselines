import numpy as np
import gym

from baselines import deepq

def MyWrapper():
    class MyWrapper(gym.ObservationWrapper):
        """
        Transforms the discrete state space into continous.
        For state i will return one-hot vector with index i
        """
        def __init__(self, env):
            super(MyWrapper, self).__init__(env)
            self.num_states = env.observation_space.n
            self.observation_space = gym.spaces.Box(0.0, 1.0, (self.num_states))

        def _observation(self, observation):
            one_hot_obs = np.zeros(self.num_states)
            one_hot_obs[observation] = 1.0
            return one_hot_obs

    return MyWrapper

def callback(lcl, glb):
    # stop training if reward exceeds 0.99
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 0.99
    return is_solved

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

def main():
    from gym.envs.toy_text import FrozenLakeEnv

    env = FrozenLakeEnv(desc=MAP, is_slippery=False)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
    wrapper = MyWrapper()
    env = wrapper(env)

    num_actions = env.action_space.n
    model = deepq.models.mlp([])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to frozen_model.pkl")
    act.save("frozen_model.pkl")


if __name__ == '__main__':
    main()

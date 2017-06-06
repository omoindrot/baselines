import time
import numpy as np
import gym
import tensorflow as tf

from baselines import logger
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


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
    # stop training if reward exceeds 0.8
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 0.8
    return is_solved


def eval_model(env, obs_placeholder, epislon_placeholder, stochastic_placeholder,
               output_actions, sess, samples=1000):
    sum_reward = 0
    for _ in range(samples):
        obs, done = env.reset(), False
        episode_rew = 0
        count = 0
        while not done:
            count += 1
            feed_dict = {obs_placeholder: np.array(obs).reshape((1,) +  obs.shape),
                         epsilon_placeholder: 0.0,
                         stochastic_placeholder: False}
            action = sess.run(output_actions, feed_dict)[0]
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        sum_reward += episode_rew
        #print(episode_rew)
        #print(count)
        #print()
    mean_reward = sum_reward / samples
    print("Mean reward over %d episodes: %f" % (samples, mean_reward))


def enjoy_model(env, act):
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(0.1)
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode rew", episode_rew)
        time.sleep(1)

#def main(args):
def main():
    tf.reset_default_graph()
    #args
    lr = 1e-3
    max_timesteps = 100000
    buffer_size = 50000
    exploration_fraction = 0.2
    exploration_final_eps = 0.02
    train_freq = 1
    batch_size = 32
    print_freq = 1
    learning_starts = 1000
    gamma = 0.99
    target_network_update_freq = 500
    grad_norm_clipping = 10

    # Create the environment
    env = gym.make("FrozenLake8x8-v0")
    wrapper = MyWrapper()
    env = wrapper(env)
    num_actions = env.action_space.n

    obs_placeholder = tf.placeholder(tf.float32, (None,) + env.observation_space.shape,
                                     name="observations")
    stochastic_placeholder = tf.placeholder(tf.bool, [], name="stochastic")
    epsilon_placeholder = tf.placeholder(tf.float32, [], name="epsilon")

    dynamic_batch_size = tf.shape(obs_placeholder)[0]

    # -------------------------------------
    # Model: gets actions with input states
    # q_values has shape (None, num_actions) and contains q(s, a) for the input batch of states
    q_values = tf.layers.dense(obs_placeholder, num_actions, name="q_values")
    deterministic_actions = tf.argmax(q_values, axis=1)

    random_actions = tf.random_uniform([dynamic_batch_size], minval=0, maxval=num_actions, dtype=tf.int64)
    chose_random = tf.random_uniform([dynamic_batch_size], minval=0, maxval=1, dtype=tf.float32) < epsilon_placeholder
    stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

    output_actions = tf.cond(stochastic_placeholder,
                             lambda: stochastic_actions,
                             lambda: deterministic_actions)

    # -------------------------------------
    # Training: given batch of s, a, r, s'
    #           update parameters
    states = tf.placeholder(tf.float32, (None,) + env.observation_space.shape, name="states")
    actions = tf.placeholder(tf.int32, [None], name="actions")
    rewards = tf.placeholder(tf.float32, [None], name="rewards")
    next_states = tf.placeholder(tf.float32, (None,) + env.observation_space.shape, name="next_states")
    done_mask = tf.placeholder(tf.float32, [None], name="done_mask")

    # q network evaluation
    q_states = tf.layers.dense(states, num_actions, reuse=True, name="q_values")

    target_q_values = tf.layers.dense(next_states, num_actions, name="target_q_values")

    # q(s,a) which were selected
    # TODO: use tf.gather_nd(q_states, tf.stack((tf.range(q_states.shape[0]), actions), 1) ?
    q_states_actions = tf.reduce_sum(q_states * tf.one_hot(actions, num_actions), 1)

    # TODO: add double q learning
    q_next_states = tf.reduce_max(target_q_values, 1)
    q_next_states_masked = q_next_states * (1.0 - done_mask)

    # Compute RHS of bellman equation
    target = rewards + gamma * q_next_states_masked

    # Compute the loss
    tf.losses.huber_loss(labels=tf.stop_gradient(target), predictions=q_states_actions)

    loss = tf.losses.get_total_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    q_network_vars = tf.contrib.framework.get_variables('q_values')
    target_q_network_vars = tf.contrib.framework.get_variables('target_q_values')
    assert(len(q_network_vars) == 2)
    assert(len(target_q_network_vars) == 2)
    # TODO: gradient clipping
    train_op = optimizer.minimize(loss, var_list=q_network_vars)

    # Update target Q network
    update_target_q_ops = []
    for var, var_target in zip(sorted(q_network_vars, key=lambda v: v.name),
                               sorted(target_q_network_vars, key=lambda v: v.name)):
        update_target_q_ops.append(var_target.assign(var))
    update_target_q = tf.group(*update_target_q_ops)

    init_op = tf.global_variables_initializer()


    # -------------------------------------
    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Create the session
    #sess = tf.Session()
    with tf.Session() as sess:

        sess.run(init_op)
        sess.run(update_target_q)

        episode_rewards = [0.0]
        obs = env.reset()

        # Save model
        # TODO: save model and stuff
        saved_mean_reward = None

        for t in range(max_timesteps):
            # Update epsilon
            epsilon = exploration.value(t)

            # Take action
            feed_dict = {obs_placeholder: np.array(obs).reshape((1,) +  obs.shape),
                         epsilon_placeholder: epsilon,
                         stochastic_placeholder: True}
            action = sess.run(output_actions, feed_dict)[0]

            new_obs, rew, done, _ = env.step(action)

            # Store transitions in the replay buffer
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)

            if t > learning_starts and t % train_freq == 0:
                experience = replay_buffer.sample(batch_size)
                obs_batch, actions_batch, rew_batch, next_obs_batch, done_mask_batch = experience
                feed_dict = {states: obs_batch,
                             actions: actions_batch,
                             rewards: rew_batch,
                             next_states: next_obs_batch,
                             done_mask: done_mask_batch}

                sess.run(train_op, feed_dict)

            if t > learning_starts and t% target_network_update_freq == 0:
                sess.run(update_target_q)

            mean_100ep_reward = np.mean(episode_rewards[-100:])
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * epsilon))
                logger.dump_tabular()

            # TODO: save regularly


    eval_model(env, obs_placeholder, epsilon_placeholder, stochastic_placeholder,
               output_actions, sess, samples=1000)


if __name__ == '__main__':
    main()

import time
import argparse
import numpy as np
import gym
import tensorflow as tf

from baselines import logger
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=1.0, help="Gamma for the agent")
parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate")
parser.add_argument('--max_timesteps', type=int, default=100000, help="Total number of timesteps")
parser.add_argument('--replay_memory', action='store_true', help="Use replay memory")
parser.add_argument('--buffer_size', type=int, default=50000, help="Buffer size for replay memory")
parser.add_argument('--initial_epsilon', type=float, default=1.0, help="Initial epsilon")
parser.add_argument('--exploration_fraction', type=float, default=0.1, help="Time spent exploring")
parser.add_argument('--final_epsilon', type=float, default=0.02, help="Final epsilon")
parser.add_argument('--train_freq', type=int, default=1, help="Train every train_freq steps")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
parser.add_argument('--print_freq', type=int, default=10, help="Print every print_freq")
parser.add_argument('--learning_starts', type=int, default=1000, help="Start training")
parser.add_argument('--target_network_update_freq', type=int, default=500, help="Update target net")
parser.add_argument('--grad_norm_clipping', type=float, default=10.0, help="Clip gradients")
parser.add_argument('--visualize', action='store_true', help="Render environment")

args = parser.parse_args()


def callback(lcl, glb):
    # stop training if reward exceeds 0.8
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def eval_model(env, obs_placeholder, epsilon_placeholder, stochastic_placeholder,
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


def enjoy_model(env, obs_placeholder, epsilon_placeholder, stochastic_placeholder,
                output_actions, sess, num_episodes=1):
    for _ in range(num_episodes):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(0.1)
            feed_dict = {obs_placeholder: np.array(obs).reshape((1,) +  obs.shape),
                         epsilon_placeholder: 0.0,
                         stochastic_placeholder: False}
            action = sess.run(output_actions, feed_dict)[0]
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode rew", episode_rew)
        time.sleep(1)


#def main(args):
#def main():
if True:
    tf.reset_default_graph()
    # Create the environment
    env = gym.make("CartPole-v0")

    num_actions = env.action_space.n

    obs_placeholder = tf.placeholder(tf.float32, (None,) + env.observation_space.shape,
                                     name="observations")
    stochastic_placeholder = tf.placeholder(tf.bool, [], name="stochastic")
    epsilon_placeholder = tf.placeholder(tf.float32, [], name="epsilon")

    dynamic_batch_size = tf.shape(obs_placeholder)[0]

    # -------------------------------------
    # Model: gets actions with input states
    # q_values has shape (None, num_actions) and contains q(s, a) for the input batch of states
    hidden_layer = tf.layers.dense(obs_placeholder, 64, name="q_values/fc1")
    q_values = tf.layers.dense(hidden_layer, num_actions, name="q_values/fc2")
    deterministic_actions = tf.argmax(q_values, axis=1)

    random_actions = tf.random_uniform([dynamic_batch_size], minval=0, maxval=num_actions,
                                       dtype=tf.int64)
    chose_random = tf.random_uniform([dynamic_batch_size], minval=0, maxval=1,
                                     dtype=tf.float32) < epsilon_placeholder
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
    next_states = tf.placeholder(tf.float32, (None,) + env.observation_space.shape,
                                 name="next_states")
    done_mask = tf.placeholder(tf.float32, [None], name="done_mask")

    # q network evaluation
    hidden_bis = tf.layers.dense(states, 64, reuse=True, name="q_values/fc1")
    q_states = tf.layers.dense(hidden_bis, num_actions, reuse=True, name="q_values/fc2")

    target_hidden = tf.layers.dense(next_states, 64, name="target_q_values/fc1")
    target_q_values = tf.layers.dense(target_hidden, num_actions, name="target_q_values/fc2")

    # q(s,a) which were selected
    # TODO: use tf.gather_nd(q_states, tf.stack((tf.range(q_states.shape[0]), actions), 1) ?
    q_states_actions = tf.reduce_sum(q_states * tf.one_hot(actions, num_actions), 1)

    # Double q learning
    # We select the best a' given the online network and estimate it with target network
    q_tp1_hidden = tf.layers.dense(next_states, 64, reuse=True, name="q_values/fc1")
    q_tp1_online = tf.layers.dense(q_tp1_hidden, num_actions, reuse=True, name="q_values/fc2")
    q_tp1_best_using_online_net = tf.arg_max(q_tp1_online, 1)
    q_next_states = tf.reduce_sum(target_q_values * tf.one_hot(q_tp1_best_using_online_net,
                                                               num_actions),
                                  axis=1)

    q_next_states_masked = q_next_states * (1.0 - done_mask)


    # -------------------------------------------------
    # Compute RHS of bellman equation
    target = rewards + args.gamma * q_next_states_masked

    # Compute the loss
    tf.losses.huber_loss(labels=tf.stop_gradient(target), predictions=q_states_actions)

    loss = tf.losses.get_total_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    q_network_vars = tf.contrib.framework.get_variables('q_values')
    target_q_network_vars = tf.contrib.framework.get_variables('target_q_values')
    assert(len(q_network_vars) == 4)
    assert(len(target_q_network_vars) == 4)

    gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=q_network_vars))
    assert(len(gradients) == 4)
    assert(len(variables) == 4)

    clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, args.grad_norm_clipping)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables))

    tf.summary.scalar("global_gradient_norm", global_norm)

    # Update target Q network
    update_target_q_ops = []
    for var, var_target in zip(sorted(q_network_vars, key=lambda v: v.name),
                               sorted(target_q_network_vars, key=lambda v: v.name)):
        update_target_q_ops.append(var_target.assign(var))
    update_target_q = tf.group(*update_target_q_ops)

    init_op = tf.global_variables_initializer()


    # -------------------------------------
    # Replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size)

    exploration = \
            LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.max_timesteps),
                           initial_p=args.initial_epsilon,
                           final_p=args.final_epsilon)

    # Create the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    #with tf.Session() as sess:
    if True:

        sess.run(init_op)
        sess.run(update_target_q)

        episode_rewards = [0.0]
        obs = env.reset()

        # Save model
        # TODO: save model and stuff
        saved_mean_reward = None

        for t in range(args.max_timesteps):
            if args.visualize:
                env.render()
                time.sleep(0.05)

            # Update epsilon
            epsilon = exploration.value(t)

            # Take action
            feed_dict = {obs_placeholder: np.array(obs).reshape((1,) +  obs.shape),
                         epsilon_placeholder: epsilon,  # TODO: hardcoded
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

            if t > args.learning_starts and t % args.train_freq == 0:
                if not args.replay_memory:
                    experience = replay_buffer.give_last(1)
                else:
                    experience = replay_buffer.sample(args.batch_size)
                obs_batch, actions_batch, rew_batch, next_obs_batch, done_mask_batch = experience
                feed_dict = {states: obs_batch,
                             actions: actions_batch,
                             rewards: rew_batch,
                             next_states: next_obs_batch,
                             done_mask: done_mask_batch}

                sess.run(train_op, feed_dict)

            if t > args.learning_starts and t% args.target_network_update_freq == 0:
                sess.run(update_target_q)

            mean_100ep_reward = np.mean(episode_rewards[-100:])
            num_episodes = len(episode_rewards)
            if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * epsilon))
                logger.dump_tabular()

            if done and args.visualize:
                time.sleep(1.0)

            # TODO: save regularly


    enjoy_model(env, obs_placeholder, epsilon_placeholder, stochastic_placeholder,
               output_actions, sess, num_episodes=1)
    print("Evaluating model")
    eval_model(env, obs_placeholder, epsilon_placeholder, stochastic_placeholder,
               output_actions, sess, samples=100)




#if __name__ == '__main__':
    #main()

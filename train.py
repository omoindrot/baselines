"""
Train a deep Q learning model on Frozen Lake.
"""

import time
import argparse
import numpy as np
import gym
import tensorflow as tf

from baselines import logger
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
from environments import EnvironmentCreator
from models import MLP, CNN_to_MLP


def eval_model(env, obs_placeholder, epsilon_placeholder, stochastic_placeholder,
               output_actions, sess, samples=1000):
    """
    Evaluate model on environement for @a samples runs.
    """
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
    """
    Render the agent on the model for @a num_episodes episodes.
    """
    for _ in range(num_episodes):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(0.02)
            feed_dict = {obs_placeholder: np.array(obs).reshape((1,) +  obs.shape),
                         epsilon_placeholder: 0.0,
                         stochastic_placeholder: False}
            action = sess.run(output_actions, feed_dict)[0]
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode rew", episode_rew)
        time.sleep(1)


def check_random_model(samples=10):
    """
    Average number of steps to get to the end with a random model.
    """
    env = gym.make("FrozenLake8x8-v0")
    total_count = 0
    for _ in range(samples):
        found = False
        count = 0
        while not found:
            _, done = env.reset(), False
            count += 1
            while not done:
                _, rew, done, _ = env.step(env.action_space.sample())
            if rew > 0:
                found = True
        total_count += count
    mean_count = total_count / float(samples)
    print("It took %d random simulations to find the reward on average" % mean_count)


def minimize_with_clipping(optimizer, loss, var_list, grad_norm_clipping=10.0):
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    clipped_grads_and_vars = [(tf.clip_by_norm(g, grad_norm_clipping), v)
                              for g, v in grads_and_vars]
    return optimizer.apply_gradients(clipped_grads_and_vars)


def main(args):
# if True:
    tf.reset_default_graph()
    # Create the environment
    env_creator = EnvironmentCreator(args.game)
    env = env_creator.get_env()

    num_actions = env.action_space.n

    obs_placeholder = tf.placeholder(tf.float32, (None,) + env.observation_space.shape,
                                     name="observations")
    stochastic_placeholder = tf.placeholder(tf.bool, [], name="stochastic")
    epsilon_placeholder = tf.placeholder(tf.float32, [], name="epsilon")

    dynamic_batch_size = tf.shape(obs_placeholder)[0]

    # -------------------------------------
    # Model: gets actions with input states
    model = MLP([])
    # q_values has shape (None, num_actions) and contains q(s, a) for the input batch of states
    q_values = model.q_function(obs_placeholder, num_actions, scope="q_values")
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
    q_states = model.q_function(states, num_actions, reuse=True, scope="q_values")

    target_q_values = model.q_function(next_states, num_actions, scope="target_q_values")

    # q(s,a) which were selected
    # TODO: use tf.gather_nd(q_states, tf.stack((tf.range(q_states.shape[0]), actions), 1) ?
    q_states_actions = tf.reduce_sum(q_states * tf.one_hot(actions, num_actions), 1)

    # Double q learning
    q_tp1_online = model.q_function(next_states, num_actions, reuse=True, scope="q_values")
    q_tp1_best_using_online_net = tf.arg_max(q_tp1_online, 1)
    q_next_states = tf.reduce_sum(target_q_values * tf.one_hot(q_tp1_best_using_online_net,
                                                               num_actions),
                                  axis=1)

    q_next_states_masked = q_next_states * (1.0 - done_mask)

    # ----------------
    # Intrinsic reward
    phi_t = tf.layers.dense(states, args.hidden_phi, name="phi/features")    # phi(t)
    phi_tp1 = tf.layers.dense(states, args.hidden_phi, reuse=True, name="phi/features")  # phi(t+1)

    #  --------------------------------------------------------
    # Inverse dynamic loss: predict a given phi(t) and phi(t+1)
    phi_t_tp1 = tf.concat([phi_t, phi_tp1], axis=1)
    relu_phi_t_tp1 = tf.nn.relu(phi_t_tp1)
    action_predictions = tf.layers.dense(relu_phi_t_tp1, num_actions, name="phi/logits")
    inverse_dynamic_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=action_predictions)
    tf.summary.scalar("inverse_dynamic_loss",inverse_dynamic_loss)
    inverse_dynamic_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    phi_vars = tf.contrib.framework.get_variables("phi")
    assert len(phi_vars) == 4
    # TODO: add gradient clipping
    inverse_dynamic_train_op = minimize_with_clipping(inverse_dynamic_optimizer,
                                                      inverse_dynamic_loss,
                                                      phi_vars,
                                                      args.grad_norm_clipping)


    # --------------------------------------------------
    # Forward model: predict phi(t+1) given phi(t) and a
    forward_input = tf.concat([tf.nn.relu(phi_t), tf.one_hot(actions, num_actions)], axis=1)
    # print(forward_input.get_shape())
    assert forward_input.get_shape().as_list() == [None, args.hidden_phi + num_actions]

    forward_pred = tf.layers.dense(forward_input, args.hidden_phi, name="forward")
    forward_loss = tf.nn.l2_loss(forward_pred - phi_tp1)
    tf.summary.scalar("forward_loss",forward_loss)

    forward_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    forward_vars = tf.contrib.framework.get_variables("forward")
    assert len(forward_vars) == 2
    # TODO: add gradient clipping
    forward_train_op = minimize_with_clipping(forward_optimizer,
                                              forward_loss,
                                              forward_vars,
                                              args.grad_norm_clipping)


    # -------------------------------------------------
    # Compute RHS of bellman equation
    intrinsic_rewards = args.eta * 0.5 * tf.reduce_sum(tf.square(forward_pred - phi_tp1), axis=1)
    tf.summary.scalar("intrinsic_reward", tf.reduce_mean(intrinsic_rewards))
    if args.curiosity:
        # TODO: get a way to store this full reward into the buffer
        rewards = rewards + intrinsic_rewards
    target = rewards + args.gamma * q_next_states_masked

    # Compute the loss
    loss = tf.losses.huber_loss(labels=tf.stop_gradient(target), predictions=q_states_actions)
    tf.summary.scalar("huber_loss", loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    q_network_vars = tf.contrib.framework.get_variables('q_values')
    # print(q_network_vars)
    target_q_network_vars = tf.contrib.framework.get_variables('target_q_values')

    # TODO: gradient clipping
    train_op = minimize_with_clipping(optimizer, loss, q_network_vars, args.grad_norm_clipping)

    # Update target Q network
    update_target_q_ops = []
    for var, var_target in zip(sorted(q_network_vars, key=lambda v: v.name),
                               sorted(target_q_network_vars, key=lambda v: v.name)):
        update_target_q_ops.append(var_target.assign(var))
    update_target_q = tf.group(*update_target_q_ops)

    init_op = tf.global_variables_initializer()

    tf.get_default_graph().finalize()


    # -------------------------------------
    # Replay buffer
    # TODO: add intrinsic reward into replaybuffer?
    # TODO: do importance sampling with this intrinsic curiosity reward?
    replay_buffer = ReplayBuffer(args.buffer_size)

    exploration = \
            LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.max_timesteps),
                           initial_p=args.initial_epsilon,
                           final_p=args.final_epsilon)

    # Create the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    merge_summary = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.run(update_target_q)

        episode_rewards = [0.0]
        obs = env.reset()

        writer = tf.summary.FileWriter(logdir=args.logdir)

        # Save model
        # TODO: save model and stuff
        #saved_mean_reward = None
        first_reward = 0 #first time agent reaches the end reward

        for it in range(args.max_timesteps):
            if env_creator.callback(it, episode_rewards):
                break

            # Update epsilon
            epsilon = exploration.value(it)

            # Take action
            feed_dict = {obs_placeholder: np.array(obs).reshape((1,) +  obs.shape),
                         epsilon_placeholder: epsilon,
                         stochastic_placeholder: True}  # no stochasticity for curiosity
            action = sess.run(output_actions, feed_dict)[0]

            new_obs, rew, done, _ = env.step(action)

            if rew > 0 and first_reward == 0:
                first_reward = it

            # Store transitions in the replay buffer
            # TODO: remove replay buffer for curiosity
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)

            if it > args.learning_starts and it % args.train_freq == 0:
                experience = replay_buffer.sample(args.batch_size)
                obs_batch, actions_batch, rew_batch, next_obs_batch, done_mask_batch = experience
                feed_dict = {states: obs_batch,
                             actions: actions_batch,
                             rewards: rew_batch,
                             next_states: next_obs_batch,
                             done_mask: done_mask_batch}

                if args.curiosity:
                    irew, _, _, _, summary_all = sess.run([intrinsic_rewards,
                                              train_op,
                                              inverse_dynamic_train_op,
                                              forward_train_op, merge_summary], feed_dict)
                else:
                    _, summary_all =  sess.run([train_op, merge_summary], feed_dict)

                writer.add_summary(summary_all)

            if it > args.learning_starts and it % args.target_network_update_freq == 0:
                sess.run(update_target_q)

            mean_100ep_reward = np.mean(episode_rewards[-100:])
            summary = tf.Summary(value=[tf.Summary.Value(tag="mean_100ep_reward", simple_value=mean_100ep_reward)])
            writer.add_summary(summary,it)
            num_episodes = len(episode_rewards)
            if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
                if args.visualize:
                    enjoy_model(env, obs_placeholder, epsilon_placeholder, stochastic_placeholder,
                                output_actions, sess, num_episodes=1)

                logger.record_tabular("steps", it)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * epsilon))
                logger.dump_tabular()

            # TODO: save regularly

        enjoy_model(env, obs_placeholder, epsilon_placeholder, stochastic_placeholder,
                    output_actions, sess, num_episodes=1)
        print("Evaluating model")
        eval_model(env, obs_placeholder, epsilon_placeholder, stochastic_placeholder,
                   output_actions, sess, samples=100)

    return first_reward




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="log/test", help="log directory")
    parser.add_argument('--game', type=str, default="frozen", help="Game to play.")
    parser.add_argument('--gamma', type=float, default=1.00, help="Gamma for the agent")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--max_timesteps', type=int, default=100000, help="Total number of timesteps")
    parser.add_argument('--buffer_size', type=int, default=50000, help="Buffer size for replay memory")
    parser.add_argument('--initial_epsilon', type=float, default=1.0, help="Initial epsilon")
    parser.add_argument('--exploration_fraction', type=float, default=0.2, help="Time spent exploring")
    parser.add_argument('--final_epsilon', type=float, default=0.02, help="Final epsilon")
    parser.add_argument('--train_freq', type=int, default=1, help="Train every train_freq steps")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--print_freq', type=int, default=100, help="Print every print_freq")
    parser.add_argument('--learning_starts', type=int, default=1000, help="Start training")
    parser.add_argument('--target_network_update_freq', type=int, default=500, help="Update target net")
    parser.add_argument('--grad_norm_clipping', type=float, default=10.0, help="Clip gradients")
    parser.add_argument('--visualize', action='store_true', help="Render environment")
    parser.add_argument('--curiosity', action='store_true', help="Activate curiosity module")
    parser.add_argument('--hidden_phi', type=int, default=16, help="Hidden dimension for phi")
    parser.add_argument('--eta', type=float, default=0.01, help="Coefficient for intrinsic reward")

    args = parser.parse_args()
    results = []
    for k in range(5):
        print(k)
        res = main(args)
        results.append(res)

    print(results)

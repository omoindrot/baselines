import time
import argparse
import numpy as np
import gym
import tensorflow as tf

from baselines import logger
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99, help="Gamma for the agent")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="learning rate")
parser.add_argument('--max_timesteps', type=int, default=2000000, help="Total number of timesteps")
parser.add_argument('--buffer_size', type=int, default=10000, help="Buffer size for replay memory")
parser.add_argument('--initial_epsilon', type=float, default=1.0, help="Initial epsilon")
parser.add_argument('--exploration_fraction', type=float, default=0.1, help="Time spent exploring")
parser.add_argument('--final_epsilon', type=float, default=0.01, help="Final epsilon")
parser.add_argument('--train_freq', type=int, default=4, help="Train every train_freq steps")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
parser.add_argument('--print_freq', type=int, default=1, help="Print every print_freq")
parser.add_argument('--learning_starts', type=int, default=10000, help="Start training")
parser.add_argument('--target_network_update_freq', type=int, default=1000, help="Update target net")
parser.add_argument('--grad_norm_clipping', type=float, default=10.0, help="Clip gradients")
parser.add_argument('--visualize', action='store_true', help="Render environment")
# TODO: add prioritized replay

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
    from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
    env = gym.make("PongNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_dqn(env))

    num_actions = env.action_space.n

    obs_placeholder = tf.placeholder(tf.float32, (None,) + env.observation_space.shape,
                                     name="observations")
    stochastic_placeholder = tf.placeholder(tf.bool, [], name="stochastic")
    epsilon_placeholder = tf.placeholder(tf.float32, [], name="epsilon")

    dynamic_batch_size = tf.shape(obs_placeholder)[0]

    # -------------------------------------
    # Model: gets actions with input states
    def q_function(states_ph, num_actions, reuse=False, scope="q_values"):
        # TODO: add dueling
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = tf.layers.conv2d(states_ph, 32, 8, 4, "same",
                                     activation=tf.nn.relu, name="conv1")
            conv2 = tf.layers.conv2d(conv1, 64, 4, 2, "same",
                                     activation=tf.nn.relu, name="conv2")
            conv3 = tf.layers.conv2d(conv2, 64, 3, 1, "same",
                                     activation=tf.nn.relu, name="conv3")
            flattened = tf.reshape(conv3, [tf.shape(conv1)[0], 11*11*64])

            # Q(s, a)
            action_hidden = tf.layers.dense(flattened, 256, tf.nn.relu, name="action/fc1")
            action_scores = tf.layers.dense(action_hidden, num_actions, name="action/fc2")
            # V(s)
            state_hidden = tf.layers.dense(flattened, 256, tf.nn.relu, name="state/fc1")
            state_score = tf.layers.dense(state_hidden, 1, name="state/fc2")
            # Q(s, a) - V(s)
            action_scores_mean = tf.reduce_mean(action_scores, axis=1, keep_dims=True)
            action_scores_centered = action_scores - action_scores_mean

            output = state_score + action_scores_centered
        return output

    # q_values has shape (None, num_actions) and contains q(s, a) for the input batch of states
    q_values = q_function(obs_placeholder, num_actions, reuse=False, scope="q_values")
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
    # TRAINING: given batch of s, a, r, s'
    #           update parameters
    states = tf.placeholder(tf.float32, (None,) + env.observation_space.shape, name="states")
    actions = tf.placeholder(tf.int32, [None], name="actions")
    rewards = tf.placeholder(tf.float32, [None], name="rewards")
    next_states = tf.placeholder(tf.float32, (None,) + env.observation_space.shape,
                                 name="next_states")
    done_mask = tf.placeholder(tf.float32, [None], name="done_mask")

    # q network evaluation
    q_states = q_function(states, num_actions, reuse=True, scope="q_values")

    target_q_values = q_function(next_states, num_actions, reuse=False, scope="target_q_values")

    # q(s,a) which were selected
    # TODO: use tf.gather_nd(q_states, tf.stack((tf.range(q_states.shape[0]), actions), 1) ?
    q_states_actions = tf.reduce_sum(q_states * tf.one_hot(actions, num_actions), 1)

    # Double q learning
    # We select the best a' given the online network and estimate it with target network
    q_tp1_online = q_function(next_states, num_actions, reuse=True, scope="q_values")
    q_tp1_best_using_online_net = tf.arg_max(q_tp1_online, 1)
    q_next_states = tf.reduce_sum(target_q_values * tf.one_hot(q_tp1_best_using_online_net,
                                                               num_actions),
                                  axis=1)

    q_next_states_masked = q_next_states * (1.0 - done_mask)


    # -------------------------------------------------
    # Compute RHS of bellman equation
    target = rewards + args.gamma * q_next_states_masked

    # Compute the loss
    #loss = tf.losses.huber_loss(labels=tf.stop_gradient(target), predictions=q_states_actions)
    diff = tf.stop_gradient(target) - q_states_actions
    errors = tf.where(tf.abs(diff) < 1.0, tf.square(diff) * 0.5, tf.abs(diff) - 0.5)
    loss = tf.reduce_mean(errors)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    q_network_vars = tf.contrib.framework.get_variables('q_values')
    target_q_network_vars = tf.contrib.framework.get_variables('target_q_values')
    assert(len(q_network_vars) == 14)
    assert(len(target_q_network_vars) == 14)

    grads_and_vars = optimizer.compute_gradients(loss, var_list=q_network_vars)

    clipped_grads_and_vars = [(tf.clip_by_norm(g, args.grad_norm_clipping), v)
                              for g, v in grads_and_vars]
    assert(len(clipped_grads_and_vars) == 14)
    train_op = optimizer.apply_gradients(clipped_grads_and_vars)

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

            if t > args.learning_starts and t % args.train_freq == 0:
                experience = replay_buffer.sample(args.batch_size)
                obs_batch, actions_batch, rew_batch, next_obs_batch, done_mask_batch = experience
                feed_dict = {states: obs_batch,
                             actions: actions_batch,
                             rewards: rew_batch,
                             next_states: next_obs_batch,
                             done_mask: done_mask_batch}

                _, loss_val = sess.run([train_op, loss], feed_dict)

            if t > args.learning_starts and t % args.target_network_update_freq == 0:
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

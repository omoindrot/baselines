"""Class for the different models for deep Q learning
"""

import tensorflow as tf


class MLP(object):
    """Multi-Layer Perceptron

    Args:
        - hiddens: list of hidden sizes
    """
    def __init__(self, hiddens=[]):
        self.hiddens = hiddens

    def q_function(self, inputs, num_actions, reuse=False, scope="q_values"):
        with tf.variable_scope(scope, reuse=reuse):
            out = inputs
            for hidden_size in self.hiddens:
                out = tf.layers.dense(out, hidden_size, tf.nn.relu, reuse=reuse)
            out = tf.layers.dense(out, num_actions, reuse=reuse)
        return out


class CNN_to_MLP(object):
    """Multi-Layer Perceptron

    Args:
        - convs: [(int, int, int)]
            list of convolutional layers in form of (num_outputs, kernel_size, stride)
        - hiddens: list of hidden sizes
        - dueling: if true, double the output MLP to compute advantage function
    """
    def __init__(self, convs=[], hiddens=[], dueling=True):
        self.convs = convs
        self.hiddens = hiddens
        self.dueling = dueling

    def q_function(self, inputs, num_actions, reuse=False, scope="q_values"):
        with tf.variable_scope(scope, reuse=reuse):
            out = inputs
            with tf.variable_scope("conv"):
                for filters, kernel, stride in self.convs:
                    out = tf.layers.conv2d(out, filters, kernel, stride, "same",
                                           activation=tf.nn.relu)

            # State V(s)
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden_size in hiddens:
                    state_out = tf.layers.dense(state_out, hidden_size, tf.nn.relu)
                state_score = tf.layers.dense(state_out, num_actions)

            # Advantage A(s,a)
            with tf.variable_score("action_value"):
                action_out = out
                for hidden_size in hiddens:
                    action_out = tf.layers.dense(action_out, hidden_size, tf.nn.relu)
                action_scores = tf.layers.dense(action_out, num_actions)

            action_scores_mean = tf.reduce_mean(action_scores, axis=1, keep_dims=True)
            action_scores_centered = action_scores - action_scores_mean

            out = state_score + action_scores_centered
        return out


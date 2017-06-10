"""Class for the different models for deep Q learning
"""

import tensorflow as tf


class MLP(object):
    """Multi-Layer Perceptron

    Args:
        - hiddens: list of hidden sizes
    """
    def __init__(self, q_hiddens, phi_hiddens, forward_hiddens, activation=tf.nn.relu):
        self.q_hiddens = q_hiddens
        self.phi_hiddens = phi_hiddens
        assert len(phi_hiddens) >= 1, "At least 1 layer required for phi"
        self.forward_hiddens = forward_hiddens
        self.activation = activation

    def q_function(self, inputs, num_actions, reuse=False, scope="q_values"):
        with tf.variable_scope(scope, reuse=reuse):
            out = inputs
            for hidden_size in self.q_hiddens:
                out = tf.layers.dense(out, hidden_size, tf.nn.relu)
            # TODO: add optimism in bias
            out = tf.layers.dense(out, num_actions)
                                  #bias_initializer=tf.random_normal_initializer(1.0, 0.1))
        return out

    def phi_function(self, inputs, reuse=False, scope="phi/features"):
        with tf.variable_scope(scope, reuse=reuse):
            out = inputs
            for hidden_size in self.phi_hiddens[:-1]:
                out = tf.layers.dense(out, hidden_size, tf.nn.relu)
            out = tf.layers.dense(out, self.phi_hiddens[-1])
            self.phi_size = out.get_shape()[-1]  #TODO: little sketchy here
        return out

    def inverse_model(self, phi_t, phi_tp1, num_actions, reuse=False, scope="phi/inverse"):
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.concat([phi_t, phi_tp1], axis=1)
            out = self.activation(out)
            out = tf.layers.dense(out, num_actions)
        return out

    def forward_model(self, phi_t, actions, num_actions, reuse=False, scope="forward"):
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.concat([phi_t, tf.one_hot(actions, num_actions)], axis=1)
            for hidden_size in self.forward_hiddens:
                out = tf.layers.dense(out, hidden_size, self.activation)
            out = tf.layers.dense(out, self.phi_size)
        return out



class CNNtoMLP(object):
    """Multi-Layer Perceptron

    Args:
        - convs: [(int, int, int)]
            list of convolutional layers in form of (num_outputs, kernel_size, stride)
        - hiddens: list of hidden sizes
        - dueling: if true, double the output MLP to compute advantage function
    """
    def __init__(self, q_convs, q_hiddens, phi_convs, phi_hiddens, forward_hiddens,
                 activation=tf.nn.relu, dueling=True):
        self.q_convs = q_convs
        self.q_hiddens = q_hiddens
        self.phi_convs = phi_convs
        self.phi_hiddens = phi_hiddens
        self.forward_hiddens = forward_hiddens
        self.activation = activation
        self.dueling = dueling  # TODO: provide the option to not use it

    def q_function(self, inputs, num_actions, reuse=False, scope="q_values"):
        """Given inputs observation, return q(s,a) for the states.

        Args:
            - inputs: batch of states
            - num_actions: number of possible actions
            - reuse: whether to reuse already created weights
            - scope: scope name for the variables in this function
        """
        with tf.variable_scope(scope, reuse=reuse):
            out = inputs
            with tf.variable_scope("conv"):
                for filters, kernel, stride in self.q_convs:
                    out = tf.layers.conv2d(out, filters, kernel, stride, "same",
                                           activation=self.activation)
            out = tf.contrib.layers.flatten(out)

            # State V(s)
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden_size in self.q_hiddens:
                    state_out = tf.layers.dense(state_out, hidden_size, self.activation)
                state_score = tf.layers.dense(state_out, num_actions)
            # TODO: add optimism in bias

            # Advantage A(s,a)
            with tf.variable_scope("action_value"):
                action_out = out
                for hidden_size in self.q_hiddens:
                    action_out = tf.layers.dense(action_out, hidden_size, self.activation)
                action_scores = tf.layers.dense(action_out, num_actions)

            action_scores_mean = tf.reduce_mean(action_scores, axis=1, keep_dims=True)
            action_scores_centered = action_scores - action_scores_mean

            out = state_score + action_scores_centered
        return out

    def phi_function(self, inputs, reuse=False, scope="phi"):
        with tf.variable_scope(scope, reuse=reuse):
            out = inputs
            for filters, kernel, stride in self.phi_convs:
                out = tf.layers.conv2d(out, filters, kernel, stride, "same",
                                       activation=self.activation)
            out = tf.contrib.layers.flatten(out)
            self.phi_size = out.get_shape()[-1]  #TODO: little sketchy here
        return out

    def inverse_model(self, phi_t, phi_tp1, num_actions, reuse=False, scope="inverse_dynamic"):
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.concat([phi_t, phi_tp1], axis=1)
            out = self.activation(out)
            for hidden_size in self.phi_hiddens:
                out = tf.layers.dense(out, hidden_size, self.activation)
            out = tf.layers.dense(out, num_actions)
        return out

    def forward_model(self, phi_t, actions, num_actions, reuse=False, scope="forward_model"):
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.concat([phi_t, tf.one_hot(actions, num_actions)], axis=1)
            for hidden_size in self.forward_hiddens:
                out = tf.layers.dense(out, hidden_size, self.activation)
            out = tf.layers.dense(out, self.phi_size)
        return out


class DoomModel(CNNtoMLP):
    def __init__(self, activation=tf.nn.elu, dueling=True):
        q_convs = [(32, 3, 2), (32, 3, 2), (32, 3, 2)]
        q_hiddens = [256]
        phi_convs = [(32, 3, 2), (32, 3, 2), (32, 3, 2), (32, 3, 2)]
        phi_hiddens = [256]
        forward_hiddens = [256]
        super(DoomModel, self).__init__(q_convs, q_hiddens, phi_convs, phi_hiddens, forward_hiddens,
                                        activation, dueling)

class PongModel(CNNtoMLP):
    def __init__(self, activation=tf.nn.relu, dueling=True):
        q_convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        q_hiddens = [256]
        phi_convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        phi_hiddens = [256]
        forward_hiddens = [256]
        super(PongModel, self).__init__(q_convs, q_hiddens, phi_convs, phi_hiddens, forward_hiddens,
                                        activation, dueling)

# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.recurrent import Recurrent

import numpy as np
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
import tensorflow as tf

class GraphConvRecurrent(Recurrent):
    """Abstract base class for convolutional recurrent layers.

    Do not use in a model -- it's not a functional layer!

    # Arguments
        units: Integer, the dimensionality of the output space
            (i.e. the number output filters in the convolution).
        graph_conv_tensor: A tensor of shape [K_adjacency_power, num_graph_nodes, num_graph_nodes],
            containing graph convolution/filter matrices.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, rocess the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.

    # Input shape
        4D tensor with shape `(num_samples, timesteps, num_nodes, input_dim)`.

    # Output shape
        - if `return_sequences`: 4D tensor with shape
            `(num_samples, timesteps, num_nodes, output_dim/units)`.
        - else, 3D tensor with shape `(num_samples, num_nodes, output_dim/units)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.
        **Note:** for the time being, masking is only supported with Theano.

    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                a `batch_input_size=(...)` to the first layer in your model.
                This is the expected shape of your inputs *including the batch
                size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    """

    def __init__(self, units,
                 graph_conv_tensor,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 **kwargs):

        super(GraphConvRecurrent, self).__init__(**kwargs)

        self.units = units  # hidden_units or dimensionality of the output space.

        self.poly_degree = graph_conv_tensor.shape[0] - 1  # adjacecny power degree
        self.num_nodes = graph_conv_tensor.shape[2]  # num nodes in a graph
        graph_conv_tensor = K.constant(graph_conv_tensor, dtype=K.floatx())
        self.graph_conv_tensor = graph_conv_tensor  # output_shape = [K, num_nodes, num_nodes]

        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.input_spec = [InputSpec(ndim=4)]
        self.state_spec = None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], input_shape[2], self.units)  # output_shape = [num_samples, timesteps, num_nodes, output_dim]
        else:
            output_shape = (input_shape[0], input_shape[2], self.units)  # output_shape = [num_samples, num_nodes, output_dim]

        if self.return_state:
            state_shape = [(input_shape[0], input_shape[2], self.units) for _ in self.states]
            return [output_shape] + state_shape  # output_shape, state_shape_hidden, state_shape_cell
        else:
            return output_shape

    def get_config(self):
        config = {'units': self.units,
                  'graph_conv_tensor': self.graph_conv_tensor,
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful}
        base_config = super(GraphConvRecurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GraphConvLSTM(GraphConvRecurrent):
    """Convolutional LSTM.

    It is similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    # Arguments
        units: Integer, the dimensionality of the output space
            (i.e. the number output filters in the convolution).
        graph_conv_tensor: A tensor of shape [K_adjacency_power, num_graph_nodes, num_graph_nodes],
            containing graph convolution/filter matrices.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, rocess the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # Input shape
        -  4D tensor with shape:
            `(num_samples, timesteps, num_nodes, input_dim)`

     # Output shape
        - if `return_sequences`
            - 4D tensor with shape:
                `(num_samples, timesteps, num_nodes, output_dim)`
        - else
            - 4D tensor with shape:
                `(num_samples, num_nodes, output_dim)`

    # Raises
        ValueError: in case of invalid constructor arguments.

    # References
        - [Structured Sequence Modeling with Graph Convolutional Recurrent Networks]
            (https://arxiv.org/abs/1612.07659)
        The current implementation does not include the feedback loop on the
        cells output
    """

    # @interfaces.legacy_GraphConvLSTM_support  # TODO: include legacy support
    def __init__(self, units,
                 graph_conv_tensor,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(GraphConvLSTM, self).__init__(units,
                                            graph_conv_tensor,
                                            return_sequences=return_sequences,
                                            go_backwards=go_backwards,
                                            stateful=stateful,
                                            **kwargs)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_spec = [InputSpec(shape=(None, self.num_nodes, self.units)),
                           InputSpec(shape=(None, self.num_nodes, self.units))]

    def build(self, input_shape):
        # input_shape = [num_samples, timesteps, num_nodes, input_dim]
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[3]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, self.num_nodes, self.input_dim))

        self.states = [None, None]  # initial states: two zero tensors of shape (num_samples, units)
        if self.stateful:
            self.reset_states()

        kernel_shape = ((self.poly_degree + 1) * self.input_dim, self.units * 4)
        self.kernel_shape = kernel_shape  # output_shape = [input_dim * K, output_dim * 4]
        recurrent_kernel_shape = ((self.poly_degree + 1) * self.units, self.units * 4)
        self.recurrent_kernel_shape = recurrent_kernel_shape  # output_shape = [output_dim * K, output_dim * 4]

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(shape=recurrent_kernel_shape,
                                                initializer=self.recurrent_initializer,
                                                name='recurrent_kernel',
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if self.unit_forget_bias:
                bias_value = np.zeros((self.units * 4,))
                bias_value[self.units: self.units * 2] = 1.
                K.set_value(self.bias, bias_value)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def get_initial_state(self, inputs):
        # get states for all-zero tensor input of shape (samples, num_nodes, output_dim)

        initial_state = K.zeros_like(inputs)  # (samples, timesteps, num_nodes, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, num_nodes, input_dim)
        shape = list(self.kernel_shape)
        shape[-1] = self.units

        initial_state = self.input_conv(initial_state, K.zeros(tuple(shape)))   # (samples, num_nodes, output_dim)
        initial_states = [initial_state for _ in range(2)]

        return initial_states

    def reset_states(self):

        if not self.stateful:
            raise RuntimeError('Layer must be stateful.')

        input_shape = self.input_spec[0].shape
        output_shape = self.compute_output_shape(input_shape)

        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size). '
                             'Got input shape: ' + str(input_shape))

        if self.return_sequences:
            num_nodes, units = output_shape[2:]
        else:
            num_nodes, units = output_shape[1:]

        if hasattr(self, 'states'):
            K.set_value(self.states[0], np.zeros((input_shape[0], num_nodes, units)))
            K.set_value(self.states[1], np.zeros((input_shape[0], num_nodes, units)))
        else:
            self.states = [K.zeros((input_shape[0], num_nodes, units)),
                           K.zeros((input_shape[0], num_nodes, units))]

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation == 0 and 0 < self.dropout < 1:
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones += 1

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.recurrent_dropout < 1:
            shape = list(self.kernel_shape)
            shape[-1] = self.units
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones = self.input_conv(ones, K.zeros(shape))
            ones += 1.

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def input_conv(self, x, w, b=None):
        # x = [num_samples, num_nodes, input_dim]
        # w = [input_dim * K, output_dim]
        # graph_conv_tensor = [K, num_nodes, num_nodes]

        conv_out = K.dot(self.graph_conv_tensor, x)  # output_shape = [K, num_nodes, num_samples, input_dim]
        conv_out = tf.transpose(conv_out, perm=[2, 1, 0, 3])  # output_shape = [num_samples, num_nodes, K, input_dim]
        conv_out_shape = conv_out.get_shape().as_list()
        conv_out = K.reshape(conv_out, shape=(-1, conv_out_shape[1], conv_out_shape[2] * conv_out_shape[3]))  # output_shape = [num_samples, num_nodes, input_dim * K]

        conv_out = K.dot(conv_out, w)  # output_shape = [num_samples, num_nodes, output_dim]

        if b is not None:
            conv_out = K.bias_add(conv_out, b)
        return conv_out  # output_shape = [num_samples, num_nodes, output_dim]

    def reccurent_conv(self, x, w):
        # x = [num_samples, num_nodes, output_dim]
        # w = [output_dim * K, output_dim]
        # graph_conv_tensor = [K, num_nodes, num_nodes]

        conv_out = K.dot(self.graph_conv_tensor, x)
        conv_out = tf.transpose(conv_out, perm=[2, 1, 0, 3])
        conv_out_shape = conv_out.get_shape().as_list()
        conv_out = K.reshape(conv_out, shape=(-1, conv_out_shape[1], conv_out_shape[2] * conv_out_shape[3]))

        conv_out = K.dot(conv_out, w)

        return conv_out  # output_shape = [num_samples, num_nodes, output_dim]

    def step(self, inputs, states):
        assert len(states) == 4
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]

        x_i = self.input_conv(inputs * dp_mask[0], self.kernel_i, self.bias_i)
        x_f = self.input_conv(inputs * dp_mask[1], self.kernel_f, self.bias_f)
        x_c = self.input_conv(inputs * dp_mask[2], self.kernel_c, self.bias_c)
        x_o = self.input_conv(inputs * dp_mask[3], self.kernel_o, self.bias_o)

        h_i = self.reccurent_conv(h_tm1 * rec_dp_mask[0], self.recurrent_kernel_i)
        h_f = self.reccurent_conv(h_tm1 * rec_dp_mask[1], self.recurrent_kernel_f)
        h_c = self.reccurent_conv(h_tm1 * rec_dp_mask[2], self.recurrent_kernel_c)
        h_o = self.reccurent_conv(h_tm1 * rec_dp_mask[3], self.recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(GraphConvLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

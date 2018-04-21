from keras import activations, initializers, constraints
from keras import regularizers
import keras.backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
import tensorflow as tf


class GraphCNN(Layer):

    def __init__(self,
                 output_dim,
                 graph_conv_filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.output_dim = output_dim

        self.num_filters = int(graph_conv_filters.shape[-2]/graph_conv_filters.shape[-1])
        self.graph_conv_filters = K.constant(graph_conv_filters)

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer.__name__ = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

        super(GraphCNN, self).__init__(**kwargs)

    def build(self, input_shape):

        self.input_dim = input_shape[-1]
        kernel_shape = (self.num_filters * self.input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=len(input_shape), axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):

        if self.input_spec.ndim == 2:
            conv_op = K.dot(self.graph_conv_filters, inputs)
            conv_op = tf.split(conv_op, self.num_filters, axis=0)
            conv_op = K.concatenate(conv_op, axis=1)
            output = K.dot(conv_op, self.kernel)

        if self.input_spec.ndim == 3:
            conv_op = K.batch_dot(self.graph_conv_filters, inputs)
            conv_op = tf.split(conv_op, self.num_filters, axis=1)
            conv_op = K.concatenate(conv_op, axis=2)
            output = K.dot(conv_op, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if self.input_spec.ndim == 2:
            return (input_shape[0], self.output_dim)
        if self.input_spec.ndim == 3:
            return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(GraphCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

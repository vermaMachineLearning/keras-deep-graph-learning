from keras import activations, initializers, constraints
from keras import regularizers
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Dropout, LeakyReLU, ELU
from keras.engine import InputSpec
import tensorflow as tf


class GraphAttentionCNN(Layer):

    def __init__(self,
                 output_dim,
                 adjacency_matrix,
                 num_filters=None,
                 graph_conv_filters=None,
                 num_attention_heads=1,
                 attention_heads_reduction='concat',  # {'concat', 'average'}
                 attention_dropout=0.5,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if attention_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        super(GraphAttentionCNN, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.adjacency_matrix = K.constant(adjacency_matrix)

        self.num_filters = num_filters
        if self.num_filters is not None:
            self.graph_conv_filters = K.constant(graph_conv_filters)

        self.num_attention_heads = num_attention_heads
        self.attention_heads_reduction = attention_heads_reduction
        self.attention_dropout = attention_dropout

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
        self.supports_masking = False

        self.input_dim = None
        self.kernels = []
        self.kernels_biases = []
        self.attention_kernels = []
        self.attention_kernels_biases = []

    def build(self, input_shape):

        self.input_dim = input_shape[-1]

        if self.num_filters is not None:
            kernel_shape = (self.num_filters * self.input_dim, self.output_dim)
        else:
            kernel_shape = (self.input_dim, self.output_dim)

        attention_kernel_shape = (2 * self.output_dim, 1)

        for _ in range(self.num_attention_heads):

            kernel = self.add_weight(shape=kernel_shape,
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)

            if self.use_bias:
                bias = self.add_weight(shape=(self.output_dim,),
                                       initializer=self.bias_initializer,
                                       name='bias',
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint)
            else:
                bias = None
            self.kernels_biases.append(bias)

            attention_kernel = self.add_weight(shape=attention_kernel_shape,
                                               initializer=self.kernel_initializer,
                                               name='attention_kernel',
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint)

            self.attention_kernels.append(attention_kernel)
            if self.use_bias:
                bias = self.add_weight(shape=(1,),
                                       initializer=self.bias_initializer,
                                       name='attention_bias',
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint)
            else:
                bias = None
            self.attention_kernels_biases.append(bias)

        self.built = True

    def graph_conv_op(self, x, kernel, bias=None):

        conv_op = K.dot(self.graph_conv_filters, x)
        conv_op = tf.split(conv_op, self.num_filters, axis=0)
        conv_op = K.concatenate(conv_op, axis=1)
        conv_out = K.dot(conv_op, kernel)

        if bias is not None:
            conv_out = K.bias_add(conv_out, bias)

        return conv_out

    def call(self, inputs):

        outputs = []

        for i in range(self.num_attention_heads):

            # conv_out = self.graph_conv_op(inputs, self.kernels[i], self.kernels_biases[i])
            if self.num_filters is not None:
                conv_out = self.graph_conv_op(inputs, self.kernels[i])
            else:
                conv_out = K.dot(inputs, self.kernels[i])

            # conv_out = K.dot(inputs, self.kernels[i])
            # if self.kernels_biases[i] is not None:
            #    conv_out = K.bias_add(conv_out, self.kernels_biases[i])

            atten_conv_out_self = K.dot(conv_out, self.attention_kernels[i][:self.output_dim])
            atten_conv_out_neigh = K.dot(conv_out, self.attention_kernels[i][self.output_dim:])

            # if self.attention_kernels_biases[i] is not None:
            #    atten_conv_out_self = K.bias_add(atten_conv_out_self, self.attention_kernels_biases[i])

            atten_coeff_matrix = atten_conv_out_self + K.transpose(atten_conv_out_neigh)
            atten_coeff_matrix = ELU(alpha=1.0)(atten_coeff_matrix)

            mask = K.exp(self.adjacency_matrix * -10e9) * -10e9
            atten_coeff_matrix = atten_coeff_matrix + mask

            atten_coeff_matrix = K.softmax(atten_coeff_matrix)
            atten_coeff_matrix = Dropout(self.attention_dropout)(atten_coeff_matrix)

            node_feature_matrix = K.dot(atten_coeff_matrix, conv_out)

            if self.attention_heads_reduction == 'concat' and self.activation is not None:
                node_feature_matrix = self.activation(node_feature_matrix)

            outputs.append(node_feature_matrix)

        if self.attention_heads_reduction == 'concat':
            output = K.concatenate(outputs)
        else:
            output = K.mean(K.stack(outputs), axis=0)
            # output = outputs[0]
            if self.activation is not None:
                output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):

        if self.attention_heads_reduction == 'concat':
            actutal_output_dim = self.output_dim * self.num_attention_heads
        else:
            actutal_output_dim = self.output_dim

        output_shape = (input_shape[0], actutal_output_dim)
        return output_shape

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
        base_config = super(GraphAttentionCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

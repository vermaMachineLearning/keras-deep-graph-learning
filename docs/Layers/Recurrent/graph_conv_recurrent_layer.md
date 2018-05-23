<span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/keras_dgl/layers/graph_convolutional_recurrent_layer.py#L116)</span>
### GraphConvLSTM

```python
GraphConvLSTM(output_dim, graph_conv_filters, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1D convolution layer (e.g. temporal convolution).

This layer creates a convolution kernel that is convolved
with the layer input over a single spatial (or temporal) dimension
to produce a tensor of outputs.
If `use_bias` is True, a bias vector is created and added to the outputs.
Finally, if `activation` is not `None`,
it is applied to the outputs as well.

When using this layer as the first layer in a model,
provide an `input_shape` argument
(tuple of integers or `None`, e.g.
`(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

__Arguments__

- __output_dim__: Positive integer, dimensionality of each graph node output space (or dimension of graph node embedding).
- __graph_conv_filters__: 3D Tensor, the dimensionality of the output space
(i.e. the number output of filters in the convolution).
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you don't specify anything, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
the output of the layer (its "activation").
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to the kernel matrix
(see [constraints](https://keras.io/constraints/)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](https://keras.io/constraints/)).

__Input shapes__

* 4D tensor with shape: `(samples, timestep, num_graph_nodes, input_dim)`<br/>

__Output shape__

* if `return_sequences`<br/>
4D tensor with shape: `(samples, timestep, num_graph_nodes, output_dim)`<br/>
* else<br />
4D tensor with shape: `(samples, num_graph_nodes, output_dim)`<br/>
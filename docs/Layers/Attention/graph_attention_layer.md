<span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/keras_dgl/layers/graph_attention_cnn_layer.py#L10)</span>
## GraphAttentionCNN

```python
GraphAttentionCNN(output_dim, adjacency_matrix, num_filters=None, graph_conv_filters=None, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

GraphAttention layer assumes a fixed input graph structure which is passed as a layer argument. As a result, the input order of graph nodes are fixed for the model and should match the nodes order in inputs. Also, graph structure can not be changed once the model is compiled. This choice enable us to use Keras Sequential API but comes with some constraints (for instance shuffling is not  possible anymore in-or-after each epoch). See further [remarks below](http://127.0.0.1:8000/Layers/Convolution/graph_conv_layer/#remarks) about this specific choice.<br />


__Arguments__

- __output_dim__: Positive integer, dimensionality of each graph node feature output space (or also referred dimension of graph node embedding).
- __adjacency_matrix__: input as a 2D tensor with shape: `(num_graph_nodes, num_graph_nodes)` with __diagonal values__ equal to 1.<br />
- __num_filters__: None or Positive integer, number of graph filters used for constructing  __graph_conv_filters__ input.
- __graph_conv_filters__: None or input as a 2D tensor with shape: `(num_filters*num_graph_nodes, num_graph_nodes)`<br />
`num_filters` is different number of graph convolution filters to be applied on graph. For instance `num_filters` could be power of graph Laplacian. Here list of graph convolutional matrices are stacked along second-last axis.<br />
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you don't specify anything, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector (recommended setting is False for this layer).
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

* 2D tensor with shape: `(num_graph_nodes, input_dim)` representing graph node input feature matrix.<br />


__Output shape__

* 2D tensor with shape: `(num_graph_nodes, output_dim)`	representing convoluted output graph node embedding (or signal) matrix.<br />


<span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/examples/graph_attention_cnn_node_classification_example.py)</span>
####__Example__: Graph Semi-Supervised Learning (or Node Label Classification)


```python
# A complete example of applying GraphCNN layer for performing node label classification.

model = Sequential()
model.add(Dropout(0.6, input_shape=(X.shape[1],)))
model.add(GraphAttentionCNN(8, 1, A, num_attention_heads=8, attention_heads_reduction='concat', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4)))
model.add(Dropout(0.6))
model.add(GraphAttentionCNN(Y.shape[1], 1, A, num_attention_heads=1, attention_heads_reduction='average', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=5e-3), metrics=['accuracy'])

NB_EPOCH = 1000

for epoch in range(1, NB_EPOCH + 1):
    model.fit(X, Y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
    Y_pred = model.predict(X, batch_size=A.shape[0])
    _, train_acc = evaluate_preds(Y_pred, [Y_train], [train_idx])
    _, test_acc = evaluate_preds(Y_pred, [Y_test], [test_idx])
    print("Epoch: {:04d}".format(epoch), "train_acc= {:.4f}".format(train_acc[0]),
          "test_acc= {:.4f}".format(test_acc[0]))

```
----


<span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/keras_dgl/layers/multi_graph_attention_cnn_layer.py#L11)</span>
## MultiGraphAttentionCNN

```python
MutliGraphCNN(output_dim, num_filters, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

MutliGraphCNN assumes that the number of nodes for each graph in the dataset is same. For graph with arbitrary size, one can simply append appropriate zero rows or columns in adjacency matrix (and node feature matrix) based on max graph size in the dataset to achieve this uniformity.

__Arguments__

- __output_dim__: Positive integer, dimensionality of each graph node feature output space (or also referred dimension of graph node embedding).
- __num_filters__: Positive integer, number of graph filters used for constructing  __graph_conv_filters__ input.
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

* __graph node feature matrix__ input as a 3D tensor with shape: `(batch_size, num_graph_nodes, input_dim)` corresponding to graph node input feature matrix for each graph.<br />
* __graph_conv_filters__ input as a 3D tensor with shape: `(batch_size, num_filters*num_graph_nodes, num_graph_nodes)` <br />
`num_filters` is different number of graph convolution filters to be applied on graph. For instance `num_filters` could be power of graph Laplacian.<br />

__Output shape__

* 3D tensor with shape: `(batch_size, num_graph_nodes, output_dim)`	representing convoluted output graph node embedding matrix for each graph in batch size.<br />



<span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/examples/multi_graph_attention_cnn_graph_classification_example.py)</span>
###__Example 3__: Graph Classification


```python
# See multi_graph_attention_cnn_graph_classification_example.py for complete code.

from keras_dgl.layers import MultiAttentionGraphCNN

X_input = Input(shape=(X.shape[1], X.shape[2]))
A_input = Input(shape=(A.shape[1], A.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))

output = MultiGraphAttentionCNN(100, num_filters=num_filters, num_attention_heads=2, attention_combine='concat', attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([X_input, A_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = MultiGraphAttentionCNN(100, num_filters=num_filters, num_attention_heads=1, attention_combine='average', attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([output, A_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = Lambda(lambda x: K.mean(x, axis=1))(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
output = Dense(Y.shape[1], activation='elu')(output)
output = Activation('softmax')(output)

nb_epochs = 500
batch_size = 169

model = Model(inputs=[X_input, A_input, graph_conv_filters_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit([X, A, graph_conv_filters], Y, batch_size=batch_size, validation_split=0.1, epochs=nb_epochs, shuffle=True, verbose=1)
```


----

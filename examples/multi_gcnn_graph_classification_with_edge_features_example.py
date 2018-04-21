from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from utils import *
from keras_dgl.layers import MultiGraphCNN

# prepare data
A = pd.read_csv('data/A_edge_matrices_mutag.csv', header=None)
A = np.array(A)
num_graph_nodes = A.shape[1]
num_graphs = 188  # hardcoded for mutag dataset

A = np.split(A, num_graphs, axis=0)
A = np.array(A)
num_edge_features = int(A.shape[1]/A.shape[2])

X = pd.read_csv('data/X_mutag.csv', header=None)
X = np.array(X)
X = np.split(X, num_graphs, axis=0)
X = np.array(X)

Y = pd.read_csv('data/Y_mutag.csv', header=None)
Y = np.array(Y)

A, X, Y = shuffle(A, X, Y)

# build graph_conv_filters
SYM_NORM = True
num_filters = num_edge_features
graph_conv_filters = preprocess_edge_adj_tensor(A, SYM_NORM)

# build model
X_input = Input(shape=(X.shape[1], X.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))

output = MultiGraphCNN(100, num_filters, activation='elu')([X_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(100, num_filters, activation='elu')([output, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = Lambda(lambda x: K.mean(x, axis=1))(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
output = Dense(Y.shape[1])(output)
output = Activation('softmax')(output)

nb_epochs = 200
batch_size = 169

model = Model(inputs=[X_input, graph_conv_filters_shape], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit([X, graph_conv_filters], Y, batch_size=batch_size, validation_split=0.1, epochs=nb_epochs, shuffle=True, verbose=1)

# sample output
# 169/169 [==============================] - 0s 90us/step - loss: 0.3578 - acc: 0.8462 - val_loss: 0.2174 - val_acc: 0.8947
# Epoch 496/500
# 169/169 [==============================] - 0s 100us/step - loss: 0.3748 - acc: 0.8521 - val_loss: 0.2179 - val_acc: 0.8947
# Epoch 497/500
# 169/169 [==============================] - 0s 113us/step - loss: 0.3656 - acc: 0.8521 - val_loss: 0.2186 - val_acc: 0.8947
# Epoch 498/500
# 169/169 [==============================] - 0s 102us/step - loss: 0.3592 - acc: 0.8462 - val_loss: 0.2178 - val_acc: 0.8947
# Epoch 499/500
# 169/169 [==============================] - 0s 102us/step - loss: 0.3746 - acc: 0.8521 - val_loss: 0.2160 - val_acc: 0.8947
# Epoch 500/500
# 169/169 [==============================] - 0s 99us/step - loss: 0.3710 - acc: 0.8580 - val_loss: 0.2152 - val_acc: 0.8947


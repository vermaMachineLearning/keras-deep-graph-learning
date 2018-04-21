from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from utils import *
from keras_dgl.layers import MultiGraphCNN

# prepare data
A = pd.read_csv('data/A_mutag.csv', header=None)
A = np.array(A)
num_graph_nodes = A.shape[1]
num_graphs = int(A.shape[0]/A.shape[1])

A = np.split(A, num_graphs, axis=0)
A = np.array(A)

X = pd.read_csv('data/X_mutag.csv', header=None)
X = np.array(X)
X = np.split(X, num_graphs, axis=0)
X = np.array(X)

Y = pd.read_csv('data/Y_mutag.csv', header=None)
Y = np.array(Y)

A, X, Y = shuffle(A, X, Y)

# build graph_conv_filters
SYM_NORM = True
num_filters = 2
graph_conv_filters = preprocess_adj_tensor_with_identity(A, SYM_NORM)

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

model = Model(inputs=[X_input, graph_conv_filters_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit([X, graph_conv_filters], Y, batch_size=batch_size, validation_split=0.1, epochs=nb_epochs, shuffle=True, verbose=1)

# sample output
# 169/169 [==============================] - 0s 66us/step - loss: 0.3526 - acc: 0.8639 - val_loss: 0.4241 - val_acc: 0.8947
# Epoch 198/200
# 169/169 [==============================] - 0s 67us/step - loss: 0.3558 - acc: 0.8580 - val_loss: 0.4261 - val_acc: 0.8947
# Epoch 199/200
# 169/169 [==============================] - 0s 62us/step - loss: 0.3555 - acc: 0.8462 - val_loss: 0.4276 - val_acc: 0.8947
# Epoch 200/200
# 169/169 [==============================] - 0s 64us/step - loss: 0.3546 - acc: 0.8521 - val_loss: 0.4273 - val_acc: 0.8947



from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
from keras.regularizers import l2
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from utils import *
from keras_dgl.layers import MultiGraphAttentionCNN

# prepare data
A = pd.read_csv('data/A_mutag.csv', header=None)
A = np.array(A)
num_graph_nodes = A.shape[1]
num_graphs = int(A.shape[0] / A.shape[1])

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

# set daigonal values to 1 in adjacency matrices
A_eye_tensor = []
for _ in range(num_graphs):
    Identity_matrix = np.eye(num_graph_nodes)
    A_eye_tensor.append(Identity_matrix)

A_eye_tensor = np.array(A_eye_tensor)
A = np.add(A, A_eye_tensor)

# build model
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

# sample output
# 169/169 [==============================] - 0s 138us/step - loss: 0.5014 - acc: 0.7633 - val_loss: 0.3162 - val_acc: 0.8436
# Epoch 496/500
# 169/169 [==============================] - 0s 127us/step - loss: 0.4770 - acc: 0.7633 - val_loss: 0.3187 - val_acc: 0.8436
# Epoch 497/500
# 169/169 [==============================] - 0s 131us/step - loss: 0.4781 - acc: 0.7574 - val_loss: 0.3196 - val_acc: 0.8436
# Epoch 498/500
# 169/169 [==============================] - 0s 120us/step - loss: 0.4925 - acc: 0.7574 - val_loss: 0.3197 - val_acc: 0.8436
# Epoch 499/500
# 169/169 [==============================] - 0s 137us/step - loss: 0.4911 - acc: 0.7692 - val_loss: 0.3161 - val_acc: 0.8436
# Epoch 500/500
# 169/169 [==============================] - 0s 127us/step - loss: 0.5004 - acc: 0.7633 - val_loss: 0.3130 - val_acc: 0.8436

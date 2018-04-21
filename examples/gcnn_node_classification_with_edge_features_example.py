from keras.layers import Dense, Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

from utils import *
from keras_dgl.layers import GraphCNN


# Prepare Data
X, A, Y = load_data(dataset='cora')
A = np.array(A.todense())

_, Y_val, _, train_idx, val_idx, test_idx, train_mask = get_splits(Y)
train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)
labels = np.argmax(Y, axis=1) + 1

# Normalize X
X /= X.sum(1).reshape(-1, 1)
X = np.array(X)

Y_train = np.zeros(Y.shape)
labels_train = np.zeros(labels.shape)
Y_train[train_idx] = Y[train_idx]
labels_train[train_idx] = labels[train_idx]

Y_test = np.zeros(Y.shape)
labels_test = np.zeros(labels.shape)
Y_test[test_idx] = Y[test_idx]
labels_test[test_idx] = labels[test_idx]

# Build Graph Convolution filters
SYM_NORM = True
A_norm = preprocess_adj_numpy(A, SYM_NORM)
num_filters = 2
graph_conv_filters = np.concatenate([A_norm, np.matmul(A_norm, A_norm)], axis=0)
graph_conv_filters = K.constant(graph_conv_filters)

# Build Model
model = Sequential()
model.add(GraphCNN(16, num_filters, graph_conv_filters, input_shape=(X.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
model.add(Dropout(0.2))
model.add(GraphCNN(Y.shape[1], num_filters, graph_conv_filters, activation='elu', kernel_regularizer=l2(5e-4)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])

nb_epochs = 500

for epoch in range(nb_epochs):
    model.fit(X, Y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
    Y_pred = model.predict(X, batch_size=A.shape[0])
    _, train_acc = evaluate_preds(Y_pred, [Y_train], [train_idx])
    _, test_acc = evaluate_preds(Y_pred, [Y_test], [test_idx])
    print("Epoch: {:04d}".format(epoch), "train_acc= {:.4f}".format(train_acc[0]), "test_acc= {:.4f}".format(test_acc[0]))

# Sample Output
# Epoch: 0495 train_acc= 0.9857 test_acc= 0.8310
# Epoch: 0496 train_acc= 0.9857 test_acc= 0.8280
# Epoch: 0497 train_acc= 0.9857 test_acc= 0.8260
# Epoch: 0498 train_acc= 0.9857 test_acc= 0.8310
# Epoch: 0499 train_acc= 0.9857 test_acc= 0.8350

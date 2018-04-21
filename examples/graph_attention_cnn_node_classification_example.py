from keras.layers import Dense, Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

from utils import *
from keras_dgl.layers import GraphAttentionCNN


# prepare data
X, A, Y = load_data_attention(dataset='cora')
A = np.array(A.todense())

_, Y_val, _, train_idx, val_idx, test_idx, train_mask = get_splits(Y)
train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)
labels = np.argmax(Y, axis=1) + 1

Y_train = np.zeros(Y.shape)
labels_train = np.zeros(labels.shape)
Y_train[train_idx] = Y[train_idx]
labels_train[train_idx] = labels[train_idx]

Y_test = np.zeros(Y.shape)
labels_test = np.zeros(labels.shape)
Y_test[test_idx] = Y[test_idx]
labels_test[test_idx] = labels[test_idx]


# build graph_conv_filters
num_filters = 3
graph_conv_filters = np.concatenate([np.eye(A.shape[0]), A, np.matmul(A, A)], axis=0)
graph_conv_filters = K.constant(graph_conv_filters)


# build model
model = Sequential()
model.add(Dropout(0.6, input_shape=(X.shape[1],)))
model.add(GraphAttentionCNN(8, A, num_filters, graph_conv_filters, num_attention_heads=8, attention_combine='concat', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4)))
model.add(Dropout(0.6))
model.add(GraphAttentionCNN(Y.shape[1], A, num_filters, graph_conv_filters, num_attention_heads=1, attention_combine='average', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])

NB_EPOCH = 1000

for epoch in range(1, NB_EPOCH + 1):
    model.fit(X, Y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
    Y_pred = model.predict(X, batch_size=A.shape[0])
    _, train_acc = evaluate_preds(Y_pred, [Y_train], [train_idx])
    _, test_acc = evaluate_preds(Y_pred, [Y_test], [test_idx])
    print("Epoch: {:04d}".format(epoch), "train_acc= {:.4f}".format(train_acc[0]), "test_acc= {:.4f}".format(test_acc[0]))

# sample output
# Epoch: 0990 train_acc= 0.9786 test_acc= 0.8470
# Epoch: 0991 train_acc= 0.9786 test_acc= 0.8470
# Epoch: 0992 train_acc= 0.9786 test_acc= 0.8460
# Epoch: 0993 train_acc= 0.9786 test_acc= 0.8440
# Epoch: 0994 train_acc= 0.9786 test_acc= 0.8430
# Epoch: 0995 train_acc= 0.9786 test_acc= 0.8440
# Epoch: 0996 train_acc= 0.9786 test_acc= 0.8410
# Epoch: 0997 train_acc= 0.9786 test_acc= 0.8410
# Epoch: 0998 train_acc= 0.9786 test_acc= 0.8410
# Epoch: 0999 train_acc= 0.9786 test_acc= 0.8400
# Epoch: 1000 train_acc= 0.9786 test_acc= 0.8420

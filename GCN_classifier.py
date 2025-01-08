import warnings , os
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras.layers import Input, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from gcn_conv import GraphConvolution
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras.models import Model, load_model
from scipy import io
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score, recall_score, precision_score
from natsort import natsorted

def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))

def F1_score(preds, labels):
    return f1_score(np.argmax(labels, 1), np.argmax(preds, 1))

def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))

def recall(preds, labels):
    return recall_score(np.argmax(labels, 1), np.argmax(preds, 1))

def precision(preds, labels):
    return precision_score(np.argmax(labels, 1), np.argmax(preds, 1))

def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()
    split_f1 = list()
    split_recall = list()
    split_precision = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))
        split_f1.append(F1_score(preds[idx_split], y_split[idx_split]))
        split_recall.append(recall(preds[idx_split], y_split[idx_split]))
        split_precision.append(precision(preds[idx_split], y_split[idx_split]))
        

    return split_loss, split_acc, split_f1, split_recall, split_precision

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_splits():
    np.random.seed(5)
    random.seed(5)
    idx = io.loadmat('./train_test_idx.mat')
    whole_idx = np.squeeze(np.asarray(idx['train_idx']) - 1)
    test_idx = np.squeeze(np.asarray(idx['test_idx']) - 1)

    a = np.arange(len(whole_idx))
    ran_idx = np.asarray(random.sample(range(0, len(whole_idx)), np.int(len(whole_idx)*0.2)))
    Aran_idx = np.setdiff1d(a, ran_idx)

    train_idx = whole_idx[Aran_idx]
    val_idx = whole_idx[ran_idx]

    whole_label = np.asarray(io.loadmat('./whole_label.mat')['GT_label'] - 1)
    one_hot_label = to_categorical(whole_label)

    y_train = np.zeros(one_hot_label.shape, dtype=np.int32)
    y_val = np.zeros(one_hot_label.shape, dtype=np.int32)
    y_test = np.zeros(one_hot_label.shape, dtype=np.int32)

    y_train[train_idx] = one_hot_label[train_idx]
    y_val[val_idx] = one_hot_label[val_idx]
    y_test[test_idx] = one_hot_label[test_idx]

    train_mask = sample_mask(train_idx, len(whole_idx) + len(test_idx))

    return y_train, y_val, y_test, train_idx, val_idx, test_idx, train_mask

def calculate_norm_G(features):
    ed_mat = np.zeros((features.shape[0], features.shape[0]))
    for i in range(ed_mat.shape[0]):
        ed_mat[i, i+1:] = np.sqrt(np.sum((features[i] - features[i+1:])**2, axis=1))

    mat = ed_mat + ed_mat.transpose()
    norm_ed_mat = 1 / (1+mat)
    return norm_ed_mat

def build_model(num_node, channel):
    X_in = Input((None, num_node))
    A_in = Input((None, None))

    graph_conv_1 = GraphConvolution(channels=channel,
                                    kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                    bias_initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                    activation=ReLU(), use_bias=True)([X_in, A_in])
    output = GraphConvolution(channels=2,
                                    kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                    bias_initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                    activation='softmax', use_bias=True)([graph_conv_1, A_in])
    model = Model(inputs=[X_in, A_in], outputs=output)
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_cross_entropy', metrics=['acc'])
    return model

y_train, y_val, y_test, train_idx, val_idx, test_idx, train_mask = get_splits()
train_idx.sort()
val_idx.sort()
test_idx.sort()

idx = io.loadmat('./train_test_idx.mat')
whole_idx = np.squeeze(np.asarray(idx['train_idx']) - 1)
test_idx = np.squeeze(np.asarray(idx['test_idx']) - 1)

whole_label = np.asarray(io.loadmat('./whole_label.mat')['GT_label'] - 1)
one_hot_label = to_categorical(whole_label)


y_test = np.zeros(one_hot_label.shape, dtype=np.int32)
y_whole = np.zeros(one_hot_label.shape, dtype=np.int32)

y_whole[whole_idx] = one_hot_label[whole_idx]
y_test[test_idx] = one_hot_label[test_idx]

meta = io.loadmat('./meta_220618.mat')['meta']
X = meta['suvr_82roi'][0][0]
G = calculate_norm_G(X)
norm_G = GraphConvolution.preprocess(G, norm=True).astype('f4')
norm_X = X/X.max()

wait = 0
preds = None
best_val_loss = np.inf
PATIENCE = 10
Epochs = 5000
import sys

num_channel = int(sys.argv[1])
update_A = sys.argv[2]

print("=============================================")
print("number of channel is {}".format(num_channel))
print("updating the similarity matirx is {}".format(update_A))
print("=============================================")

model = build_model(82, num_channel)
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr=0.01))


for epoch in range(1, Epochs+1):
    model.fit([norm_X, norm_G], y_train, sample_weight=train_mask, batch_size = norm_G.shape[0], epochs=1,
              shuffle=False, verbose=0)
    
    preds = model.predict([norm_X, norm_G], batch_size = norm_G.shape[0])
    
    train_val_loss, train_val_acc, _, _, _ = evaluate_preds(preds, [y_train, y_val],
                                                   [train_idx, val_idx])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.10f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "Patience= {}".format(wait),
          "best validation loss= {:.10f}".format(best_val_loss))
    
    if update_A == "True":
        if (epoch%10) == 0:
             W = model.layers[2].weights[0].numpy()
             tmp_X = np.dot(X, W)
             G = calculate_norm_G(tmp_X)
             norm_G = GraphConvolution.preprocess(G, norm=True).astype('f4')
             print('similarity matrix updated')
    
    if epoch > 500:
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1
model.save('Amlyoid_classification_channel_{}_update_A_{}.h5'.format(num_channel, update_A))
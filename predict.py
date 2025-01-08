import warnings , os
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import pickle
from tensorflow.keras.layers import Input, ReLU, Softmax, Dropout
from tensorflow.keras.models import Model, load_model
from gcn_conv import GraphConvolution
from scipy import io
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score
from natsort import natsorted
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))

def F1_score(preds, labels):
    return f1_score(np.argmax(labels, 1), np.argmax(preds, 1))

def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()
    split_f1 = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))
        split_f1.append(F1_score(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc, split_f1

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

y_train, y_val, y_test, train_idx, val_idx, test_idx, train_mask = get_splits()

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

files = os.listdir('./no_update_221018/')
models = []
for i in files:
    if i.endswith('.h5'):
        models.append(i)
models = natsorted(models)
print(models)

loss = []

with tf.device('/cpu:0'):
    for n, i in enumerate(models):
        model = load_model('./no_update_221018/{}'.format(i), custom_objects={"GraphConvolution": GraphConvolution, "Softmax": Softmax, "ReLU":ReLU})
        W = model.layers[2].weights[0].numpy()
        tmp_X = np.dot(X, W)
        G = calculate_norm_G(tmp_X)
        norm_G = GraphConvolution.preprocess(G, norm=True).astype('f4')
        preds = model.predict([norm_X, norm_G], batch_size=G.shape[0])
        test_loss, test_acc, test_f1 = evaluate_preds(preds, [y_val], [val_idx])
        loss.append(test_loss[0])
        print("model = {}, Acc = {:.6f}, Loss = {:.6f}, F1 = {:.6f}".format(i[:-3], test_acc[0], test_loss[0], test_f1[0]))


i = np.argmin(loss)
print(models[i])
with tf.device('/cpu:0'):
    model = load_model('./no_update_221018/{}'.format(models[i]), custom_objects={"GraphConvolution": GraphConvolution, "Softmax": Softmax, "ReLU":ReLU})
    model.summary()
    W = model.layers[2].weights[0].numpy()
    tmp_X = np.dot(X, W)
    G = calculate_norm_G(tmp_X)
    norm_G = GraphConvolution.preprocess(G, norm=True).astype('f4')
    preds = model.predict([norm_X, norm_G], batch_size=G.shape[0])
    test_loss, test_acc, test_f1 = evaluate_preds(preds, [y_whole, y_test], [whole_idx, test_idx])
    print("Acc = {:.6f}, Loss = {:.6f}, F1 = {:.6f}".format(test_acc[0], test_loss[0], test_f1[0]))
    print("Test Acc = {:.6f}, Loss = {:.6f}, F1 = {:.6f}".format(test_acc[1], test_loss[1], test_f1[1]))


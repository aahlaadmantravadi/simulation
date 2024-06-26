#!/usr/bin/env python2.7

import numpy as np
import networkx as nx
from scipy.io import loadmat
import scipy.sparse as sp
from itertools import combinations
import tensorflow as tf


np.random.seed(1982)


def generate_data(adj, adj_train, feats, y_true, mask, shuffle=True):
    if shuffle:
        indices = np.arange(adj.shape[0])
        np.random.shuffle(indices)
    else:
        indices = range(adj.shape[0])

    for i in indices:
        a = adj[i].toarray()
        t = adj_train[i].toarray()
        # Convert the sparse tensor to a dense tensor
        f_dense = tf.sparse.to_dense(feats[i])
        # Convert the dense tensor to a NumPy array
        f = f_dense.numpy()
        yield (a, t, f, y_true[i], mask[i])





def batch_data(data, batch_size):
    while True:  # this flag yields an infinite generator
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(data))
            except StopIteration:
                break
        if len(batch) > 0:
            a, t, f, y, m = zip(*batch)
            a = np.vstack(a)
            t = np.vstack(t)
            f = np.vstack(f)
            y = np.vstack(y)
            m = np.vstack(m)
            yield map(np.float32, (a, t, f, y, m))
        else:
            break



def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    from keras import backend as K
    lrate = base_lr * (1.0 - (curr_iter / float(max_iter)))**power
    K.set_value(model.optimizer.lr, lrate)
    return K.eval(model.optimizer.lr)


def load_mat_data(dataset_str):
    """ dataset_str: protein, metabolic, conflict, powergrid """
    dataset_path = 'data/' + dataset_str + '.mat'
    mat = loadmat(dataset_path)
    if dataset_str == 'powergrid':
        adj = sp.lil_matrix(mat['G'], dtype=np.float32)
        feats = None
        return adj, feats
    adj = sp.lil_matrix(mat['D'], dtype=np.float32)
    feats = sp.lil_matrix(mat['F'].T, dtype=np.float32)
    # Return matrices in scipy sparse linked list format
    return adj, feats
        

def split_train_test(dataset_str, adj, fold=0, ratio=0.0):
    assert fold in range(10), 'Choose fold in range [0,9]'
    upper_inds = [ind for ind in combinations(range(adj.shape[0]), r=2)]
    np.random.shuffle(upper_inds)
    if dataset_str in ['arxiv-grqc', 'blogcatalog']:
        split = int(ratio * len(upper_inds))
        return np.asarray(upper_inds[:split])
    test_inds = []
    for ind in upper_inds:
        rand = np.random.randint(0, 10)
        if dataset_str == 'powergrid':
        # Select 10% of {powergrid} dataset for test split
            boolean = (rand == fold)
        else:
        # Select 90% of {protein, metabolic, conflict} datasets
        # for test split
            boolean = (rand != fold)
        if boolean:
            test_inds.append(ind)
    return np.asarray(test_inds)


def compute_masked_accuracy(y_true, y_pred, mask):
    correct_preds = np.equal(np.argmax(y_true, 1), np.argmax(y_pred, 1))
    num_examples = float(np.sum(mask))
    correct_preds *= mask
    return np.sum(correct_preds) / num_examples


def compute_precisionK(adj, reconstruction, K):
    """ Modified from https://github.com/suanrong/SDNE """
    N = adj.shape[0]
    reconstruction = reconstruction.reshape(-1)
    sortedInd = np.argsort(reconstruction)[::-1]
    curr = 0
    count = 0
    precisionK = []
    for ind in sortedInd:
        x = ind / N
        y = ind % N
        count += 1
        if (adj[x, y] == 1 or x == y):
            curr += 1
        precisionK.append(1.0 * curr / count)
        if count >= K:
            break
    return precisionK


def create_adj_from_edgelist(dataset_str):
    """ dataset_str: arxiv-grqc, blogcatalog """
    dataset_path = 'data/' + dataset_str + '.txt'
    with open(dataset_path, 'r') as f:
        header = next(f)
        edgelist = []
        for line in f:
            i,j = map(int, line.split())
            edgelist.append((i,j))
    g = nx.Graph(edgelist)
    return sp.lil_matrix(nx.adjacency_matrix(g))


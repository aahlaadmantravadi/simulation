#!/usr/bin/env python3
"""
This script trains a model based on the symmetrical autoencoder
architecture with parameter sharing. The model performs reconstruction
using only latent features learned from local graph topology.

Usage: python train_reconstruction.py <dataset_str> <gpu_id>
"""
import itertools
import sys
if len(sys.argv) < 3:
    print('\nUSAGE: python %s <dataset_str> <gpu_id>' % sys.argv[0])
    sys.exit()
dataset = sys.argv[1]
gpu_id = sys.argv[2]

import numpy as np
import scipy.sparse as sp
from tensorflow.keras import backend as K

import tensorflow as tf

from utils import create_adj_from_edgelist, compute_precisionK
from utils import generate_data, batch_data, split_train_test
from longae.models.ae import autoencoder

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

print('\nLoading dataset {:s}...\n'.format(dataset))

try:
    adj = create_adj_from_edgelist(dataset)
except IOError:
    sys.exit('Supported strings: {arxiv-grqc, blogcatalog}')

original = adj.copy()
train = adj.copy()
missing_edges = split_train_test(dataset, adj, ratio=0.0)
if len(missing_edges) > 0:
    r = missing_edges[:, 0]
    c = missing_edges[:, 1]
    train[r, c] = -1.0
    train[c, r] = -1.0
    adj[r, c] = 0.0
    adj[c, r] = 0.0

# Initialize the features matrix with Glorot uniform initializer
feats = tf.Variable(initial_value=tf.keras.initializers.GlorotUniform()((adj.shape[0], 128)), trainable=False)

print('\nCompiling autoencoder model...\n')
encoder, ae = autoencoder(dataset, adj)
print(ae.summary())

# Define the train_on_batch method with the tf.function decorator
@tf.function(reduce_retracing=True)
def train_step(batch_adj, batch_train):
    # Explicitly cast the batch_adj and batch_train tensors to float64
    batch_adj = tf.cast(batch_adj, tf.float64)
    batch_train = tf.cast(batch_train, tf.float64)

    with tf.GradientTape() as tape:
        loss = ae([batch_adj], training=True)
    gradients = tape.gradient(loss, ae.trainable_variables)
    ae.optimizer.apply_gradients(zip(gradients, ae.trainable_variables))
    return loss

# Define a scalar tensor variable to accumulate the training loss
train_loss_tensor = tf.Variable(0.0, trainable=False)

# Specify some hyperparameters
epochs = 50
train_batch_size = 32
val_batch_size = 256

print('\nFitting autoencoder model...\n')

train_data = generate_data(adj, train, feats.to_tensor(), y_true, mask, shuffle=True)

batch_data = batch_data(train_data, train_batch_size)
num_iters_per_train_epoch = adj.shape[0] / train_batch_size
for e in range(epochs):
    print('\nEpoch {:d}/{:d}'.format(e+1, epochs))
    print('Learning rate: {:.6f}'.format(ae.optimizer.learning_rate.numpy()))

    # Reset the train_loss_tensor variable at the beginning of each epoch
    train_loss_tensor.assign(0.0)

    curr_iter = 0
    for batch_adj, batch_train, dummy_f, dummy_y, dummy_m in batch_data:
        # Each iteration/loop is a batch of train_batch_size samples
        res = train_step(batch_adj, batch_train)

        # Accumulate the training loss in the train_loss_tensor variable
        train_loss_tensor.assign_add(res)

        curr_iter += 1
        if curr_iter >= num_iters_per_train_epoch:
            break

    # Convert the train_loss_tensor variable to a numpy array and compute the mean
    train_loss = train_loss_tensor.numpy()
    train_loss = np.mean(train_loss)

    print('Avg. training loss: {:6f}'.format(train_loss))

print('\nEvaluating reconstruction performance...')
reconstruction = np.empty(shape=adj.shape, dtype=np.float32)
for step in range(adj.shape[0] / val_batch_size + 1):
    low = step * val_batch_size
    high = low + val_batch_size
    batch_adj = adj[low:high].toarray()
    if batch_adj.shape[0] == 0:
        break
    reconstruction[low:high] = ae.predict_on_batch([batch_adj])
print('Computing precision@k...')
k = [10, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
if dataset == 'blogcatalog':
    k = k[:1] + [item*10 for item in k[1:]]
precisionK = compute_precisionK(original, reconstruction, np.max(k))
for index in k:
    if index == 0:
        index += 1
    print('Precision@{:d}: {:6f}'.format(index, precisionK[index-1]))
print('\nAll Done.')
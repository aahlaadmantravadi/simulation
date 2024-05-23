import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def mvn(tensor):
    """Per row mean-variance normalization."""
    epsilon = 1e-6
    mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    return mvn

def mbce(y_true, y_pred):
    """ Masked binary cross-entropy loss """
    mask = tf.math.not_equal(y_true, -1.0)
    mask = tf.cast(mask, dtype=tf.float32)
    num_examples = tf.math.reduce_sum(mask, axis=1)
    pos = tf.cast(tf.math.equal(y_true, 1.0), dtype=tf.float32)
    num_pos = tf.math.reduce_sum(pos, axis=None)
    neg = tf.cast(tf.math.equal(y_true, 0.0), dtype=tf.float32)
    num_neg = tf.math.reduce_sum(neg, axis=None)
    pos_ratio = 1.0 - num_pos / num_neg
    mbce = mask * tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred,
            pos_weight=pos_ratio
    )
    mbce = tf.math.reduce_sum(mbce, axis=1) / num_examples
    return tf.math.reduce_mean(mbce, axis=-1)


def ce(y_true, y_pred):
    """ Binary cross-entropy loss """
    return tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true), axis=-1)

def masked_categorical_crossentropy(y_true, y_pred):
    """ Masked categorical cross-entropy loss """
    mask = y_true[:, -1]
    y_true = y_true[:, :-1]
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
    mask = tf.cast(mask, dtype=tf.float32)
    loss *= mask
    return tf.math.reduce_mean(loss, axis=-1)

class DenseTied(tf.keras.layers.Layer):
    def __init__(self, units, tie_to, transpose=False, activation=None, **kwargs):
        super(DenseTied, self).__init__(**kwargs)
        self.units = units
        self.tie_to = tie_to
        self.transpose = transpose
        self.activation = activation

    def build(self, input_shape):
    # Create a trainable weight variable for this layer.
        self.kernel = self.tie_to.kernel
        if self.transpose:
            self.kernel = tf.transpose(self.kernel)
        # Create a trainable bias variable for this layer.
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    name='bias',
                                    trainable=True)
        super(DenseTied, self).build(input_shape)  # Be sure to call this at the end


    def call(self, inputs):
        output = tf.keras.backend.dot(inputs, self.kernel) + self.bias
        if self.activation is not None:
            output = tf.nn.relu(output)  # for ReLU activation
        return output




def autoencoder(dataset, adj, weights=None):
    h, w = adj.shape
    sparse_net = dataset in ['conflict', 'metabolic', 'protein']

    kwargs = dict(
        use_bias=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        trainable=True,
    )

    data = Input(shape=(w,), dtype=tf.float32, name='data')
    if sparse_net:
        # for conflict, metabolic, protein networks
        noisy_data = Dropout(rate=0.2, name='drop0')(data)
    else:
        # for citation, blogcatalog, arxiv-grqc, and powergrid networks
        noisy_data = Dropout(rate=0.5, name='drop0')(data)

    ### First set of encoding transformation ###
    encoded = Dense(256, activation='relu',
            name='encoded1', **kwargs)(noisy_data)
    if sparse_net:
        encoded = Lambda(mvn, name='mvn1')(encoded)
        encoded = Dropout(rate=0.5, name='drop1')(encoded)

    ### Second set of encoding transformation ###
    encoded = Dense(128, activation='relu',
            name='encoded2', **kwargs)(encoded)
    if sparse_net:
        encoded = Lambda(mvn, name='mvn2')(encoded)
    encoded = Dropout(rate=0.5, name='drop2')(encoded)

    # the encoder model maps an input to its encoded representation
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')

    ### First set of decoding transformation ###
    decoded = DenseTied(256, tie_to=encoded2, transpose=True,
            activation='relu', name='decoded2')(encoded)
    if sparse_net:
        decoded = Lambda(mvn, name='mvn3')(decoded)
        decoded = Dropout(rate=0.5, name='drop3')(decoded)

    ### Second set of decoding transformation - reconstruction ###
    decoded = DenseTied(w, tie_to=encoded1, transpose=True,
            activation='linear', name='decoded1')(decoded)

    # compile the autoencoder
    adam = Adam(learning_rate=0.001, decay=0.0)
    autoencoder = Model(inputs=[data], outputs=[decoded])
    autoencoder.compile(optimizer=adam, loss=mbce)

    if weights is not None:
        autoencoder.load_weights(weights)

    return encoder, autoencoder

def autoencoder_with_node_features(dataset, adj, feats, weights=None):
    aug_adj = sp.hstack([adj, feats])
    h, w = aug_adj.shape

    kwargs = dict(
        use_bias=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        trainable=True,
    )

    data = Input(shape=(w,), dtype=tf.float32, name='data')
    if dataset in ['protein', 'cora', 'citeseer', 'pubmed']:
        # dropout 0.5 is needed for protein and citation (large nets)
        noisy_data = Dropout(rate=0.5, name='drop1')(data)
    else:
        # dropout 0.2 is needed for conflict and metabolic (small nets)
        noisy_data = Dropout(rate=0.2, name='drop1')(data)

    ### First set of encoding transformation ###
    encoded = Dense(256, activation='relu',
            name='encoded1', **kwargs)(noisy_data)
    encoded = Lambda(mvn, name='mvn1')(encoded)

    ### Second set of encoding transformation ###
    encoded = Dense(128, activation='relu',
            name='encoded2', **kwargs)(encoded)
    encoded = Lambda(mvn, name='mvn2')(encoded)
    encoded = Dropout(rate=0.5, name='drop2')(encoded)

    # the encoder model maps an input to its encoded representation
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')

    ### First set of decoding transformation ###
    decoded = DenseTied(256, tie_to=encoded2, transpose=True,
            activation='relu', name='decoded2')(encoded)
    decoded = Lambda(mvn, name='mvn3')(decoded)

    ### Second set of decoding transformation - reconstruction ###
    decoded = DenseTied(w, tie_to=encoded1, transpose=True,
            activation='linear', name='decoded1')(decoded)

    # compile the autoencoder
    adam = Adam(lr=0.001, decay=0.0)
    if dataset in ['metabolic', 'conflict']:
        # datasets with mixture of binary and real features
        decoded_feats = Lambda(lambda x: x[:, adj.shape[1]:],
                            name='decoded_feats')(decoded)
        decoded = Lambda(lambda x: x[:, :adj.shape[1]],
                            name='decoded')(decoded)
        autoencoder = Model(
                inputs=[data], outputs=[decoded, decoded_feats]
        )
        autoencoder.compile(
                optimizer=adam,
                loss={'decoded': mbce, 'decoded_feats': ce},
                loss_weights={'decoded': 1.0, 'decoded_feats': 1.0}
        )
    else:
        # datasets with only binary graph and node features
        autoencoder = Model(inputs=[data], outputs=decoded)
        autoencoder.compile(optimizer=adam, loss=mbce)

    if weights is not None:
        autoencoder.load_weights(weights)

    return encoder, autoencoder

def autoencoder_multitask(dataset, adj, feats, labels, weights=None):
    adj = sp.hstack([adj, feats])
    h, w = adj.shape

    kwargs = dict(
        use_bias=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        trainable=True,
    )

    data = Input(shape=(w,), dtype=tf.float32, name='data')

    ### First set of encoding transformation ###
    encoded = Dense(256, activation='relu',
            name='encoded1', **kwargs)(data)

    ### Second set of encoding transformation ###
    encoded = Dense(128, activation='relu',
            name='encoded2', **kwargs)(encoded)
    if dataset == 'pubmed':
        encoded = Dropout(rate=0.5, name='drop')(encoded)
    else:
        encoded = Dropout(rate=0.8, name='drop')(encoded)

    # the encoder model maps an input to its encoded representation
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')

    ### First set of decoding transformation ###
    decoded = DenseTied(256, tie_to=encoded2, transpose=True,
            activation='relu', name='decoded2')(encoded)

    ### Node classification ###
    feat_data = Input(shape=(feats.shape[1],))
    pred1 = Dense(labels.shape[1], activation='linear')(feat_data)
    pred2 = Dense(labels.shape[1], activation='linear')(decoded)
    prediction = Add([pred1, pred2], name='prediction')

    ### Second set of decoding transformation - reconstruction ###
    decoded = DenseTied(w, tie_to=encoded1, transpose=True,
            activation='linear', name='decoded1')(decoded)

    # compile the autoencoder
    adam = Adam(lr=0.001, decay=0.0)
    autoencoder = Model(inputs=[data, feat_data],
                        outputs=[decoded, prediction])
    autoencoder.compile(
            optimizer=adam,
            loss={'decoded1': mbce,
                  'prediction': masked_categorical_crossentropy},
            loss_weights={'decoded1': 1.0, 'prediction': 1.0}
    )

    if weights is not None:
        autoencoder.load_weights(weights)

    return encoder, autoencoder

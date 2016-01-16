#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from keras.preprocessing import sequence, text
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from keras.utils import np_utils

import utils

import sys
import logging


def build_graph(graph, embedding_size=100, embedding_path=None, token2idx=None,
                input_dropout_rate=0.25, dropout_rate=0.5, l1=None, l2=None,
                convolutional_kernels=16, filter_extensions=[3, 4, 5], fix_embeddings=False,
                max_features=100000, max_len=100, output_dim=80):
    '''
    Builds Keras Graph model that, given a query (in the form of a list of indices), returns a vector of output_dim
    non-negative weights that sum up to 1.
    The Convolutional Neural Network architecture is inspired by the following paper:
    Yoon Kim - Convolutional Neural Networks for Sentence Classification - arXiv:1408.5882v2
    '''
    regularizer = utils.get_regularizer(l1, l2)

    graph.add_input(name='input_query', input_shape=(None,), dtype='int32')

    E = None
    if embedding_path is not None:
        E = utils.read_embeddings(embedding_path, token2idx=token2idx, max_features=max_features)

    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_size, input_length=max_len, weights=E)

    if fix_embeddings is True:
        embedding_layer.params = []
        embedding_layer.updates = []

    graph.add_node(embedding_layer, name='embedding', input='input_query')

    graph.add_node(Dropout(input_dropout_rate), name='embedding_dropout', input='embedding')

    flatten_layer_names = []
    for w_size in filter_extensions:
        convolutional_layer = Convolution1D(input_dim=embedding_size, nb_filter=convolutional_kernels,
                                            filter_length=w_size, border_mode='valid', activation='relu',
                                            W_regularizer=regularizer, subsample_length=1)

        convolutional_layer_name = 'convolutional' + str(w_size)
        graph.add_node(convolutional_layer, name=convolutional_layer_name , input='embedding_dropout')

        pool_length = convolutional_layer.output_shape[1]
        pooling_layer = MaxPooling1D(pool_length=pool_length)

        pooling_layer_name = 'pooling' + str(w_size)
        graph.add_node(pooling_layer, name=pooling_layer_name, input=convolutional_layer_name)

        flatten_layer_name = 'flatten' + str(w_size)
        flatten_layer = Flatten()
        graph.add_node(flatten_layer, name=flatten_layer_name, input=pooling_layer_name)
        flatten_layer_names += [flatten_layer_name]

    graph.add_node(Dropout(dropout_rate), name='dropout', inputs=flatten_layer_names, merge_mode='concat')

    dense_layer = Dense(output_dim=output_dim, W_regularizer=regularizer)
    graph.add_node(dense_layer, name='dense', input='dropout')

    softmax_layer = Activation('softmax')
    graph.add_node(softmax_layer, name='softmax', input='dense')

    return graph


def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def main(argv):
    max_features = 100000
    max_len = 100

    lines = read_lines(argv[0])

    labels, questions = [], []
    for line in lines:
        tokens = line.split()

        labels += [tokens[0]]
        questions += [' '.join(tokens[1:])]

    # Train the tokenizer on both training and validation sets
    tokenizer = text.Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(questions)

    sequences = [seq for seq in tokenizer.texts_to_sequences_generator(questions)]

    X = sequence.pad_sequences(sequences, maxlen=max_len)

    label2idx = {label: idx for idx, label in enumerate(sorted(set(labels)), 0)}
    nb_classes = len(label2idx)

    labels_idx = [label2idx[label] for label in labels]
    y = np_utils.to_categorical(labels_idx, nb_classes)

    logging.info('X is: %s' % str(X.shape))
    logging.info('y is: %s' % str(y.shape))

    graph = Graph()

    graph = build_graph(graph, l2=1e-4)
    graph.add_output(name='output', input='softmax')

    graph.compile(optimizer='adadelta', loss={'output': 'categorical_crossentropy'})

    graph.fit({'input_query': X, 'output': y}, epochs=100)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])

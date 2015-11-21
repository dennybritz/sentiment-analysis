#! /usr/bin/env python

import sys
import os
import tensorflow as tf
from models.base import BaseNN

class CharCNN(BaseNN):
    def __init__(self, vocabulary_size, embedding_size=128, filter_sizes=[1, 2, 3], num_filters=128,
                 affine_dim=256, dropout_keep_prob=0.5, num_gpus=1, optimizer=None):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.affine_dim = affine_dim
        self.dropout_keep_prob = dropout_keep_prob
        self.num_gpus = num_gpus
        self.optimizer = optimizer if optimizer else tf.train.AdamOptimizer(1e-4)

    def build_conv_maxpool(self, filter_shape, pool_shape, input_tensor):
        """
        Builds a convolutional layer followed by a max-pooling layer.
        """
        W = tf.get_variable("W", filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", filter_shape[-1], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding="VALID")
        h = tf.nn.relu(conv + b, name="conv")
        return tf.nn.max_pool(h, ksize=pool_shape, strides=[1, 1, 1, 1], padding='VALID', name="pool")

    def inference(self, x, labels):
        """
        Builds the graph and returns the final prediction.
        """

        sequence_length = x.get_shape().as_list()[1]
        num_classes = labels.get_shape().as_list()[1]

        with tf.variable_scope("embedding"):
            embedded_chars = self.build_embedding_layer([self.vocabulary_size, self.embedding_size], x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                with tf.device("/gpu:%d" % (i % self.num_gpus)):
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    pool_shape = [1, sequence_length - filter_size + 1, 1, 1]
                    pooled = self.build_conv_maxpool(filter_shape, pool_shape, embedded_chars_expanded)
                    pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Affine Layer with dropout
        with tf.variable_scope("affine"):
            h_affine = self.build_affine([num_filters_total, self.affine_dim], h_pool_flat)
        h_affine_drop = tf.nn.dropout(h_affine, self.dropout_keep_prob)

        # Softmax Layer (Final output)
        with tf.variable_scope("softmax"):
            y = self.build_softmax([self.affine_dim, num_classes], h_affine_drop)

        # Return final prediction
        return y

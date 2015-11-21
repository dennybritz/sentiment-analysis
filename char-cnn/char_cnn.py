#! /usr/bin/env python
import sys
import os
import tensorflow as tf
from utils.mixins import NNMixin, TrainMixin


class CharCNN(object, NNMixin, TrainMixin):
    """
    A CNN for text classifications
    Embedding -> Convolutinal Layer -> Affine Layer -> Softmax Prediction
    """
    def __init__(
        self, vocabulary_size, sequence_length, num_classes=2, embedding_size=128,
            filter_sizes=[1, 2, 3], num_filters=128, affine_dim=256, dropout_keep_prob=0.5, num_gpus=1):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length])
        self.input_y = tf.placeholder(tf.float32, [None, num_classes])

        self.embedded_chars = self._build_embedding([vocabulary_size, embedding_size], self.input_x)
        # Add another dimension, expected by the convolutional layer
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size), tf.device("/gpu:%d" % (i % num_gpus)):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                pool_filter_shape = [1, sequence_length - filter_size + 1, 1, 1]
                pooled = self._build_conv_maxpool(filter_shape, pool_filter_shape, self.embedded_chars_expanded)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Affine Layer with dropout
        self.h_affine = self._build_affine([num_filters_total, affine_dim], self.h_pool_flat)
        self.h_affine_drop = tf.nn.dropout(self.h_affine, dropout_keep_prob)

        # Softmax Layer (Final output)
        self.y = self._build_softmax([affine_dim, num_classes], self.h_affine_drop)

        # Loss
        self.loss = self._build_total_ce_loss(self.y, self.input_y)
        self.mean_loss = self._build_mean_ce_loss(self.y, self.input_y)

        # Summaries
        total_loss_summary = tf.scalar_summary("total loss", self.loss)
        mean_loss_summary = tf.scalar_summary("mean loss", self.mean_loss)
        accuracy_summmary = tf.scalar_summary("accuracy", self._build_accuracy(self.y, self.input_y))
        self.summaries = tf.merge_all_summaries()

    def _build_conv_maxpool(self, filter_shape, pool_shape, input_tensor):
        """
        Builds a convolutional layer followed by a max-pooling layer.
        """
        W = tf.get_variable("W", filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", filter_shape[-1], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding="VALID")
        h = tf.nn.relu(conv + b, name="conv")
        return tf.nn.max_pool(h, ksize=pool_shape, strides=[1, 1, 1, 1], padding='VALID', name="pool")

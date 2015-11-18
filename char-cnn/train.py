#! /usr/bin/env python

import sys, os
sys.path.append(os.pardir)
import numpy as np
import tensorflow as tf
import utils.ymr_data as ymr
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import math

# Hyperparameters
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
SENTENCE_LENGTH_PADDED= int(os.getenv("SENTENCE_LENGTH_PADDED", "512"))
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
L1_NUM_FILTERS = int(os.getenv("L1_NUM_FILTERS", "50"))
TRAIN_SUMMARY_DIR = os.getenv("TRAIN_SUMMARY_DIR", "./summaries/train") 
DEV_SUMMARY_DIR = os.getenv("TRAIN_SUMMARY_DIR", "./summaries/dev")
EVALUATE_DEV_EVERY = int(os.getenv("EVALUATE_DEV_EVERY", "128"))

PADDING_CHARACTER =  u"\u0000"
NUM_CLASSES=6

# Data Preparation
# ==================================================
df = ymr.load()

# Preprocessing: Pad all sentences
df.text = df.text.str.slice(0,SENTENCE_LENGTH_PADDED).str.ljust(SENTENCE_LENGTH_PADDED, PADDING_CHARACTER)

# Generate vocabulary and dataset
vocab, vocab_inv = ymr.vocab(df)
data = ymr.make_polar(df)
train, test = ymr.train_test_split(data)
train_x, train_y_ = ymr.make_xy(train, vocab)
test_x, test_y_ = ymr.make_xy(test, vocab)

VOCABULARY_SIZE = len(vocab)

# Convert ys to probability distribution
train_y = np.zeros((len(train_y_), NUM_CLASSES))
train_y[np.arange(len(train_y_)), train_y_] = 1.
test_y = np.zeros((len(test_y_), NUM_CLASSES))
test_y[np.arange(len(test_y_)), test_y_] = 1.

# Use a dev set
train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.05)

# Generate batches
num_batch_arrays = int(len(train_x)/BATCH_SIZE)
train_x_batched = np.array_split(train_x, num_batch_arrays)
train_y_batched = np.array_split(train_y, num_batch_arrays)

print("Training set size: %d" % len(train_y))
print("Dev set size: %d" % len(dev_y))
print("Test set size: %d" % len(test_y))


# Build the graph
# ==================================================

# Network inputs and output
x = tf.placeholder(tf.int32, shape=[None, SENTENCE_LENGTH_PADDED], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name="y")

# Variables
W_embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0), name="W_embeddings")

# Layer 1: Embedding
embed = tf.nn.embedding_lookup(W_embeddings, x)
# Add a dimension corresponding to the channel - it's expected by the conv layer
embed_expanded = tf.expand_dims(embed, -1)
embed_shape = tf.shape(embed_expanded)

# Layer 2: Simple Convolutional Layer
W_conv1 = tf.Variable(tf.truncated_normal([3, EMBEDDING_SIZE, 1, L1_NUM_FILTERS], stddev=0.1), name="W_conv1")
b_conv1 = tf.Variable(tf.constant(0.1, shape=[L1_NUM_FILTERS]), name="b_conv1")
h_conv1_tmp = tf.nn.conv2d(embed_expanded, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(h_conv1_tmp + b_conv1)
h_conv1_shape = tf.shape(h_conv1_tmp)

# Layer 3: Max-pooling
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, SENTENCE_LENGTH_PADDED, EMBEDDING_SIZE, 1], strides=[1, 1, 1, 1], padding='VALID')
h_pool1_shape = tf.shape(h_pool1)

# Layer 4: Fully connected
h_pool1_flat = tf.reshape(h_pool1, [-1, L1_NUM_FILTERS])
h_pool1_flat_shape = tf.shape(h_pool1_flat)
W_fc1 = tf.Variable(tf.truncated_normal([L1_NUM_FILTERS, 256], stddev=0.1), name="W_fc1")
b_fc1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b_fc1")
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
h_fc1_shape = tf.shape(h_fc1)

# TODO: Dropout?

# Layer 5: Softmax / Readout
W_fc2 = tf.Variable(tf.truncated_normal([256, NUM_CLASSES], stddev=0.1), name="W_fc2")
b_fc2 =  tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
y_conv_shape = tf.shape(y_conv)

# Training procedure
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv), name="crossentropy_sum")
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

# Summaries
ce_summary = tf.scalar_summary("cross-entropy", cross_entropy)
accuracy_summary = tf.scalar_summary("accuracy", accuracy)
summary_op = tf.merge_all_summaries()


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    summary_writer_train = tf.train.SummaryWriter(TRAIN_SUMMARY_DIR, graph_def=sess.graph_def)
    summary_writer_dev = tf.train.SummaryWriter(DEV_SUMMARY_DIR, graph_def=sess.graph_def)
    step = 0
    for epoch in range(NUM_EPOCHS):
        for i in range(len(train_x_batched)):
            feed_dict = { x: train_x_batched[i], y_ : train_y_batched[i]}
            _, loss = sess.run([train_step, cross_entropy], feed_dict=feed_dict)
            print("step %d, train loss: %g" % (step, loss))
            train_summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer_train.add_summary(train_summary_str, step)
            if(step % EVALUATE_DEV_EVERY == 0):
                feed_dict = { x: dev_x[:64], y_ :dev_y[:64]}
                dev_loss, dev_accuracy, dev_summary_str = sess.run([cross_entropy, accuracy, summary_op], feed_dict=feed_dict)
                summary_writer_dev.add_summary(dev_summary_str, step)
                print "step %d, dev loss %g"%(step, dev_loss)
                print "step %d, dev accuracy %g"%(step, dev_accuracy)
            step += 1

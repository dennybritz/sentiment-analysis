#! /usr/bin/env python

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
import time
from sklearn.cross_validation import train_test_split

sys.path.append(os.pardir)
import utils.ymr_data as ymr

# Parameters
# ==================================================

# Model Hyperparameters
SENTENCE_LENGTH_PADDED = int(os.getenv("SENTENCE_LENGTH_PADDED", "512"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "128"))
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", "128"))

# Training parameters
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
EVALUATE_EVERY = int(os.getenv("EVALUATE_EVERY", "16"))

# Output files
RUNDIR = "./runs/%s" % int(time.time())
CHECKPOINT_PREFIX = os.getenv("CHECKPOINT_PREFIX", "%s/checkpoints/char-cnn" % RUNDIR)
TRAIN_SUMMARY_DIR = os.getenv("TRAIN_SUMMARY_DIR", "%s/summaries/train" % RUNDIR)
DEV_SUMMARY_DIR = os.getenv("TRAIN_SUMMARY_DIR", "%s/summaries/dev" % RUNDIR)

# Misc Parameters
NUM_GPUS = int(os.getenv("NUM_GPUS", "4"))
ALLOW_SOFT_PLACEMENT = bool(os.getenv("ALLOW_SOFT_PLACEMENT", 1))
LOG_DEVICE_PLACEMENT = bool(os.getenv("LOG_DEVICE_PLACEMENT", 0))
PADDING_CHARACTER = u"\u0000"
NUM_CLASSES = 2

if not os.path.exists(os.path.dirname(CHECKPOINT_PREFIX)):
    os.makedirs(os.path.dirname(CHECKPOINT_PREFIX))


# Data Preparation
# ==================================================
df = ymr.load()

# Preprocessing: Pad all sentences
df.text = df.text.str.slice(0, SENTENCE_LENGTH_PADDED).str.ljust(
    SENTENCE_LENGTH_PADDED, PADDING_CHARACTER)

# Generate vocabulary and dataset
vocab, vocab_inv = ymr.vocab(df)
data = ymr.make_polar(df)
train, test = ymr.train_test_split(data)
train_x, train_y_ = ymr.make_xy(train, vocab)
test_x, test_y_ = ymr.make_xy(test, vocab)

VOCABULARY_SIZE = len(vocab)

# Convert ys to one-hot vectors (probability distribution)
train_y = np.zeros((len(train_y_), NUM_CLASSES))
train_y[np.arange(len(train_y_)), train_y_] = 1.
test_y = np.zeros((len(test_y_), NUM_CLASSES))
test_y[np.arange(len(test_y_)), test_y_] = 1.

# Use a dev set
train_x, dev_x, train_y, dev_y = train_test_split(
    train_x, train_y, test_size=0.05)

# Print data sizes
print("\nData Size")
print("----------")
print("Training set size: %d" % (len(train_y)))
print("Dev set size: %d" % len(dev_y))
print("Test set size: %d" % len(test_y))


# Build the graph
# ==================================================
# Keeps track of shapes, for debugging purposes
shape_tensors = []

def debug_shape(name, tensor):
    full_name = "%s-shape" % name
    shape_tensors.append(tf.shape(tensor, name=full_name))

# Input data
# --------------------------------------------------

# Store the data in graph notes
train_x_const = tf.constant(train_x.astype("int32"))
train_y_const = tf.constant(train_y.astype("float32"))
# Use Tensorflow's queues and batching features
x_slice, y_slice = tf.train.slice_input_producer(
    [train_x_const, train_y_const],
    num_epochs=NUM_EPOCHS)
x, y_ = tf.train.batch([x_slice, y_slice], batch_size=BATCH_SIZE)


# Layer 1: Embedding
# --------------------------------------------------
# Not supported by GPU...
with tf.device('/cpu:0'):
    with tf.name_scope("embedding"):
        W_embeddings = tf.Variable(
            tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0),
            name="W")
        embed = tf.nn.embedding_lookup(W_embeddings, x)
        # Add a dimension corresponding to the channel - it's expected by the conv
        # layer
        embed_expanded = tf.expand_dims(embed, -1)
        debug_shape("W", embed_expanded)

# RNN Layer
with tf.name_scope("lstm"):
    lstm_cell = rnn_cell.BasicLSTMCell(HIDDEN_DIM, forget_bias=0.0)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * 5)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, states = rnn.rnn(cell, x, initial_state=initial_state)
    debug_shape("outputs", outputs)
    debug_shape("states", states)

# output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_DIM])
# last_output = output[-1]


def print_shapes():
    """
    Prints the shapes of the graph for one batch
    """
    sess = tf.get_default_session()
    feed_dict = {x: train_x[:1], y_: train_y[:1]}
    shapes = sess.run(shape_tensors, feed_dict=feed_dict)
    print("\nShapes")
    print("----------")
    for k, v in zip(shape_tensors, shapes):
        print("%s: %s" % (k.name, v))

# Training Loop
# ==================================================

# Print parameters
print "\nParameters:"
print("----------")
total_parameters = 0
for v in tf.trainable_variables():
    num_parameters = v.get_shape().num_elements()
    print("{}: {:,}".format(v.name, num_parameters))
    total_parameters += num_parameters
print("\nTotal Parameters: {:,}\n".format(total_parameters))

# Write graph
tf.train.write_graph(default_graph_def, "%s/graph" % RUNDIR, "graph.pb", as_text=False)

# Initialize training
step = 0

session_config = tf.ConfigProto(
    log_device_placement=LOG_DEVICE_PLACEMENT,
    allow_soft_placement=ALLOW_SOFT_PLACEMENT)

with tf.Session(config=session_config) as sess:
    sess.run(tf.initialize_all_variables())
    # Initialize queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print_shapes()

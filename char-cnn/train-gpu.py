#! /usr/bin/env python

import sys
import os
import numpy as np
import tensorflow as tf
import math
from sklearn import metrics
from sklearn.cross_validation import train_test_split

sys.path.append(os.pardir)
import utils.ymr_data as ymr


# Parameters
# ==================================================

# Model Hyperparameters
SENTENCE_LENGTH_PADDED = int(os.getenv("SENTENCE_LENGTH_PADDED", "512"))
AFFINE_LAYER_DIM = int(os.getenv("AFFINE_LAYER_DIM", "256"))
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", "256"))
L1_NUM_FILTERS = int(os.getenv("L1_NUM_FILTERS", "128"))
L1_FILTER_SIZES = map(int, os.getenv("L1_FILTER_SIZES", "1,2,3,4").split(","))

# Training parameters
NUM_GPUS = int(os.getenv("NUM_GPUS", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
DROPOUT_PROB = float(os.getenv("DROPOUT_PROB", "0.5"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
EVALUATE_EVERY = int(os.getenv("EVALUATE_EVERY", "16"))

# Output files
CHECKPOINT_PREFIX = os.getenv("CHECKPOINT_PREFIX", "./checkpoints/char-cnn")
TRAIN_SUMMARY_DIR = os.getenv("TRAIN_SUMMARY_DIR", "./summaries/train")
DEV_SUMMARY_DIR = os.getenv("TRAIN_SUMMARY_DIR", "./summaries/dev")

# Misc Parameters
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

# Device placement for GPU
# Unsupported ops run on CPU, matmul on GPU
# def device_for_node(n):
#     if n.type == "MatMul":
#         return "/gpu:0"
#     else:
#         return "/cpu:0"


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
# Not supported by GPU?
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


# Layer 2 and 3: Convolution + Max-pooling
# --------------------------------------------------
def build_convpool(filter_size, num_filters):
    """
    Helper function to build a convolution + max-pooling layer
    """
    W = tf.get_variable(
        "weights",
        [filter_size, EMBEDDING_SIZE, 1, num_filters],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable(
        "bias",
        [num_filters],
        initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(
        embed_expanded, W, strides=[1, 1, 1, 1], padding='VALID')
    h_conv = tf.nn.relu(conv + b, name="conv")
    return tf.nn.max_pool(
        h_conv,
        ksize=[1, SENTENCE_LENGTH_PADDED - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")

# For each filter size, build a convolution + maxpool layer
pooled_outputs = []
for i, filter_size in L1_FILTER_SIZES:
    # Put each conv layer on a separate GPU if possible
    with tf.device("/gpu:%d" % (i % NUM_GPUS)):
        with tf.variable_scope("conv-maxpool-%s" % filter_size):
            pooled = build_convpool(filter_size, L1_NUM_FILTERS)
            pooled_outputs.append(pooled)
            debug_shape("h", pooled)

# Combine all the pooled features into one tensor
h_pool = tf.concat(3, pooled_outputs)
debug_shape("pooled-output-final-h", h_pool)

# Layer 4: Fully connected (affine) layer
# --------------------------------------------------
total_filters = L1_NUM_FILTERS * len(L1_FILTER_SIZES)
with tf.name_scope("affine"):
    # Flatten the pooled features into a [batch, features] vector
    h_pool_flat = tf.reshape(h_pool, [-1, total_filters])
    W_affine = tf.Variable(
        tf.truncated_normal([total_filters, AFFINE_LAYER_DIM], stddev=0.1),
        name="W_affine")
    b_affine = tf.Variable(
        tf.constant(0.1, shape=[AFFINE_LAYER_DIM]),
        name="b_affine")
    h_affine = tf.nn.relu(
        tf.matmul(h_pool_flat, W_affine) + b_affine,
        name="h_affine")
    debug_shape("pooled-flattened", h_pool_flat)
    debug_shape("h", h_affine)

# Dropout
# --------------------------------------------------
with tf.name_scope("dropout"):
    h_affine_drop = tf.nn.dropout(h_affine, DROPOUT_PROB)

# Layer 5: Softmax / Readout
# --------------------------------------------------
with tf.name_scope("softmax"):
    W_softmax = tf.Variable(tf.truncated_normal(
        [AFFINE_LAYER_DIM, NUM_CLASSES], stddev=0.1), name="W")
    b_softmax = tf.Variable(
        tf.constant(0.1, shape=[NUM_CLASSES]), name="b")
    y = tf.nn.softmax(
        tf.matmul(h_affine_drop, W_softmax) + b_softmax, name="y")

# Training procedure
# --------------------------------------------------
with tf.name_scope("loss"):
    ce_loss_mean = -tf.reduce_mean(y_ * tf.log(y), name="ce_loss_mean")
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(
        tf.argmax(y, 1), tf.argmax(y_, 1), name="correct_predictions")
    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, "float"), name="accuracy")
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(ce_loss_mean)

# Summaries
# --------------------------------------------------
ce_summary = tf.scalar_summary("cross-entropy loss", ce_loss_mean)
accuracy_summary = tf.scalar_summary("accuracy", accuracy)
summary_op = tf.merge_all_summaries()


# Training
# ==================================================

default_graph_def = tf.get_default_graph().as_graph_def()
summary_writer_train = tf.train.SummaryWriter(
    TRAIN_SUMMARY_DIR, graph_def=default_graph_def)
summary_writer_dev = tf.train.SummaryWriter(
    DEV_SUMMARY_DIR, graph_def=default_graph_def)
saver = tf.train.Saver()


def train_batch(step):
    """
    Trains a single batch
    """
    sess = tf.get_default_session()
    # feed_dict = {x: batch_x, y_: batch_y}
    _, train_loss, train_accuracy, train_summary_str = sess.run(
        [train_step, ce_loss_mean, accuracy, summary_op])
    summary_writer_train.add_summary(train_summary_str, step)


def evaluate_dev(step):
    """
    Evaluates loss and accuracy on the dev data
    """
    sess = tf.get_default_session()
    feed_dict = {x: dev_x, y_: dev_y}
    dev_loss, dev_accuracy, dev_summary_str = sess.run(
        [ce_loss_mean, accuracy, summary_op], feed_dict=feed_dict)
    print "step %d, dev loss %g" % (step, dev_loss)
    print "step %d, dev accuracy %g" % (step, dev_accuracy)
    summary_writer_dev.add_summary(dev_summary_str, step)


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

# Initialize training
step = 0

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.initialize_all_variables())
    # Initialize queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print_shapes()
    print("\nTraining Start")
    print("----------")
    # Train until we're done
    try:
        while not coord.should_stop():
            train_batch(step)
            if(step % EVALUATE_EVERY == 0):
                evaluate_dev(step)
                saver.save(sess, CHECKPOINT_PREFIX, global_step=step)
            step += 1
    except tf.errors.OutOfRangeError:
        # We're done
        saver.save(sess, CHECKPOINT_PREFIX, global_step=step)
        print("Done. Exiting.")
    finally:
        coord.request_stop()
    # Clean up
    coord.join(threads)

#! /usr/bin/env python

import numpy as np
import os
import sys
import tensorflow as tf
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

sys.path.append(os.pardir)

from char_cnn import CharCNN
from utils import ymr_data


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("sentence_length", 256, "Padded sentence length")
tf.flags.DEFINE_integer("affine_layer_dim", 256, "Dimensionality of affine (fully-connected) layer")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters, per filter size")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", 128, "Number of training epochs")
tf.flags.DEFINE_integer("batch_size", 128, "Input data batch size")
tf.flags.DEFINE_integer("evaluate_every", 16, "Evaluate model on dev set after this number of steps")

# Misc Parameters
tf.flags.DEFINE_integer("num_gpus", 4, "Max number of GPUs to use")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow soft device placement (e.g. no GPU)")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Get data
train_x, train_y, dev_x, dev_y, test_x, test_y = ymr_data.generate_dataset(fixed_length=FLAGS.sentence_length)
vocabulary_size = max(train_x.max(), dev_x.max(), test_x.max()) + 1
print("\ntrain/dev/test size: {:d}/{:d}/{:d}\n".format(len(train_y), len(dev_y), len(test_y)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # Instantiate our model
        cnn = CharCNN(
            vocabulary_size,
            FLAGS.sentence_length,
            2,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            affine_dim=FLAGS.affine_layer_dim,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            num_gpus=FLAGS.num_gpus)

        # Generate input batches (using tensorflow)
        with tf.variable_scope("input"):
            placeholder_x = tf.placeholder(tf.int32, train_x.shape)
            placeholder_y = tf.placeholder(tf.float32, train_y.shape)
            train_x_var = tf.Variable(placeholder_x, trainable=False, collections=[])
            train_y_var = tf.Variable(placeholder_y, trainable=False, collections=[])
            x_slice, y_slice = tf.train.slice_input_producer([train_x_var, train_y_var], num_epochs=FLAGS.num_epochs)
            x_batch, y_batch = tf.train.batch([x_slice, y_slice], batch_size=FLAGS.batch_size)

        # Define Training procedure
        out_dir = os.path.join(os.path.curdir, "runs", str(int(time.time())))
        global_step = tf.Variable(0, name="global_step")
        train_op = tf.train.AdamOptimizer(1e-4).minimize(cnn.loss, global_step=global_step)

        # Generate train and eval seps
        train_step = cnn.build_train_step(out_dir, train_op, global_step, cnn.summaries, save_every=8, sess=sess)
        eval_step = cnn.build_eval_step(out_dir, cnn.predictions, global_step, cnn.summaries, sess=sess)

        # Initialize variables and input data
        sess.run(tf.initialize_all_variables())
        sess.run([train_x_var.initializer, train_y_var.initializer], {placeholder_x: train_x, placeholder_y: train_y})

        # Initialize queues
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Print model parameters
        cnn.print_parameters()

        # Repeat until we're done (the input queue throws an error)...
        try:
            while not coord.should_stop():
                train_step({cnn.input_x: x_batch.eval(), cnn.input_y: y_batch.eval()})
                if global_step.eval() % FLAGS.evaluate_every == 0:
                    predictions, _, _ = eval_step({cnn.input_x: dev_x, cnn.input_y: dev_y})
                    print(classification_report(np.argmax(dev_y, axis=1), predictions))

        except tf.errors.OutOfRangeError:
            print("Yay, training done!")
            eval_step({cnn.input_x: dev_x, cnn.input_y: dev_y})
        finally:
            coord.request_stop()
        coord.join(threads)

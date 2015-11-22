#! /usr/bin/env python

import numpy as np
import os
import sys
import tensorflow as tf
import time

from sklearn.cross_validation import train_test_split

sys.path.append(os.pardir)

from char_rnn import CharRNN
from utils import ymr_data


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("sentence_length", 256, "Padded sentence length")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_integer("hidden_dim", 256, "Dimensionality of the cell")
tf.flags.DEFINE_integer("num_layers", 3, "Number of stacked layers in the RNN cell")

# Training parameters
tf.flags.DEFINE_string("loss_type", "linear_gain", "One of last or linear_gain")
tf.flags.DEFINE_integer("num_epochs", 128, "Number of training epochs")
tf.flags.DEFINE_integer("batch_size", 128, "Input data batch size")
tf.flags.DEFINE_integer("evaluate_every", 16, "Evaluate model on dev set after this number of steps")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow soft device placement (e.g. no GPU)")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
dir(FLAGS)
for attr, value in FLAGS.__flags.iteritems():
    print("{}: {}".format(attr, value))

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
        rnn = CharRNN(
            vocabulary_size,
            FLAGS.sentence_length,
            FLAGS.batch_size,
            2,
            embedding_size=FLAGS.embedding_dim,
            hidden_dim=FLAGS.hidden_dim,
            num_layers=FLAGS.num_layers,
            loss=FLAGS.loss_type)

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
        optimizer = tf.train.AdamOptimizer(1e-4)
        # Clip the gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(rnn.loss, tvars), 5)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

        # Generate train and eval seps
        train_step = rnn.build_train_step(out_dir, train_op, global_step, rnn.summaries, save_every=8, sess=sess)
        eval_step = rnn.build_eval_step(out_dir, global_step, rnn.summaries, sess=sess)

        # Initialize variables and input data
        sess.run(tf.initialize_all_variables())
        sess.run([train_x_var.initializer, train_y_var.initializer], {placeholder_x: train_x, placeholder_y: train_y})

        # Initialize queues
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Print model parameters
        # rnn.print_parameters()

        # Repeat until we're done (the input queue throws an error)...
        try:
            while not coord.should_stop():
                train_step({rnn.input_x: x_batch.eval(), rnn.input_y: y_batch.eval()})
                if global_step.eval() % FLAGS.evaluate_every == 0:
                    eval_step({rnn.input_x: dev_x[:FLAGS.batch_size], rnn.input_y: dev_y[:FLAGS.batch_size]})
        except tf.errors.OutOfRangeError:
            print("Yay, training done!")
            eval_step({rnn.input_x: dev_x, rnn.input_y: dev_y})
        finally:
            coord.request_stop()
        coord.join(threads)

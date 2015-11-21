#! /usr/bin/env python

import sys
import os
import numpy as np
import tensorflow as tf
import time
from sklearn.cross_validation import train_test_split

sys.path.append(os.pardir)

from char_cnn import CharCNN
from utils import ymr_data
from models.trainer import Trainer


# Parameters
# ==================================================

# Model Hyperparameters
SENTENCE_LENGTH_PADDED = int(os.getenv("SENTENCE_LENGTH_PADDED", "256"))
AFFINE_LAYER_DIM = int(os.getenv("AFFINE_LAYER_DIM", "256"))
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", "128"))
NUM_FILTERS = int(os.getenv("NUM_FILTERS", "128"))
FILTER_SIZES = map(int, os.getenv("FILTER_SIZES", "1,2,3").split(","))
DROPOUT_PROB = float(os.getenv("DROPOUT_PROB", "0.5"))

# Training parameters
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
EVALUATE_EVERY = int(os.getenv("EVALUATE_EVERY", "16"))

# Misc Parameters
NUM_GPUS = int(os.getenv("NUM_GPUS", "4"))
ALLOW_SOFT_PLACEMENT = bool(os.getenv("ALLOW_SOFT_PLACEMENT", 1))
LOG_DEVICE_PLACEMENT = bool(os.getenv("LOG_DEVICE_PLACEMENT", 0))

# Get data
train_x, train_y, test_x, test_y = ymr_data.generate_dataset(fixed_length=SENTENCE_LENGTH_PADDED)
train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.05)
VOCABULARY_SIZE = max(train_x.max(), dev_x.max(), test_x.max())
print("\ntrain/dev/test size: {:d}/{:d}/{:d}\n".format(len(train_y), len(dev_y), len(test_y)))


with tf.Graph().as_default():
    # Out model
    cnn = CharCNN(
        VOCABULARY_SIZE,
        embedding_size=EMBEDDING_SIZE,
        filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS,
        affine_dim=AFFINE_LAYER_DIM,
        dropout_keep_prob=DROPOUT_PROB,
        num_gpus=NUM_GPUS)

    # Generate input batches
    with tf.variable_scope("input"):
        placeholder_x = tf.placeholder(tf.int32, train_x.shape)
        placeholder_y = tf.placeholder(tf.float32, train_y.shape)
        train_x_var = tf.Variable(placeholder_x, trainable=False, collections=[])
        train_y_var = tf.Variable(placeholder_y, trainable=False, collections=[])
        x, labels = cnn.build_input_batches(train_x_var, train_y_var, NUM_EPOCHS, BATCH_SIZE)

    # Generate predictions
    predictions = cnn.inference(x, labels)

    # Loss
    with tf.variable_scope("loss"):
        loss = cnn.loss(predictions, labels)

    # Train
    global_step = tf.Variable(0, name="global_step")
    train_op = cnn.train(loss, global_step)

    # Summaries
    with tf.variable_scope("metrics"):
        tf.scalar_summary("loss", loss)
        tf.scalar_summary("accuracy", cnn.accuracy(predictions, labels))
        summary_op = tf.merge_all_summaries()

    # Create anew session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=ALLOW_SOFT_PLACEMENT))
    with sess.as_default():
        # Creat a train helper
        eval_feed_dict = {x: dev_x, labels: dev_y}
        trainer = Trainer(
            train_op, global_step, summary_op, eval_feed_dict, evaluate_every=EVALUATE_EVERY,
            save_every=EVALUATE_EVERY)
        # Initialize Variables and input data
        sess.run(
            [tf.initialize_all_variables(), train_x_var.initializer, train_y_var.initializer],
            {placeholder_x: train_x, placeholder_y: train_y})
        # Initialize queues
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # Print model parameters
        cnn.print_parameters()
        try:
            while not coord.should_stop():
                trainer.step()
        except tf.errors.OutOfRangeError:
            print("Done!")
        finally:
            coord.request_stop()
        coord.join(threads)

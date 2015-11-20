#! /usr/bin/env python

import sys
import os
import numpy as np
import tensorflow as tf
import math
import time
import distutils.util
from sklearn import metrics
from sklearn.cross_validation import train_test_split

sys.path.append(os.pardir)
import utils.ymr_data as ymr

GRAPH_DEF_FILE = sys.argv[1]
CHECKPOINT_FILE = sys.argv[2]

# df = ymr.load()

# # Preprocessing: Pad all sentences
# df.text = df.text.str.slice(0, SENTENCE_LENGTH_PADDED).str.ljust(
#     SENTENCE_LENGTH_PADDED, PADDING_CHARACTER)

# # Generate vocabulary and dataset
# vocab, vocab_inv = ymr.vocab(df)
# data = ymr.make_polar(df)
# train, test = ymr.train_test_split(data)
# test_x, test_y_ = ymr.make_xy(test, vocab)

# # Convert ys to one-hot vectors (probability distribution)
# test_y = np.zeros((len(test_y_), NUM_CLASSES))
# test_y[np.arange(len(test_y_)), test_y_] = 1.


# Build the graph
# ==================================================

# Import graph def
with open(GRAPH_DEF_FILE, 'rb') as f:
  graph_def = tf.GraphDef.FromString(f.read())
  tf.import_graph_def(graph_def)
tf.get_default_graph()


with tf.Session() as sess:
  # Restore checkpoint
  saver = tf.train.Saver()
  saver.restore(sess, CHECKPOINT_FILE)
  print "Model restored"
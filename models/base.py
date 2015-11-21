import tensorflow as tf


class BaseNN:
    def __init__(self, vocabulary_size, embedding_size=128, optimizer=None):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.optimizer = optimizer if optimizer else tf.train.AdamOptimizer(1e-4)

    def build_input_batches(self, train_x, train_y, num_epochs, batch_size):
        """
        Builds a graph that stores the input in memory and uses queues
        to slice it into bactches.

        Returns a node representing batches of x and y.
        """
        # Use Tensorflow's queues and batching features
        x_slice, y_slice = tf.train.slice_input_producer([train_x, train_y], num_epochs=num_epochs)
        x_batch, y_batch = tf.train.batch([x_slice, y_slice], batch_size=batch_size)
        return [x_batch, y_batch]

    def build_embedding_layer(self, shape, input_tensor):
        """
        Builds an embedding layer.

        Returns the final embedding.
        """
        # We force this on the CPU because the op isn't implemented for the GPU yet
        with tf.device('/cpu:0'):
            W_intializer = tf.random_uniform(shape, -1.0, 1.0)
            W_embeddings = tf.Variable(W_intializer, name="W")
            return tf.nn.embedding_lookup(W_embeddings, input_tensor)

    def build_affine(self, shape, input_tensor):
        """
        Builds an affine (fully-connected) layer
        """
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name="b")
        h = tf.nn.relu(tf.matmul(input_tensor, W) + b, name="h")
        return h

    def build_softmax(self, shape, input_tensor):
        """
        Builds a softmax layer
        """
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name="b")
        return tf.nn.softmax(tf.matmul(input_tensor, W) + b, name="y")

    def loss(self, predictions, labels):
        """
        Calculates the mean cross-entropy loss
        """
        return -tf.reduce_mean(labels * tf.log(predictions), name="loss")

    def accuracy(self, y, labels):
        """
        Returns accuracy tensor
        """
        correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1), name="correct_predictions")
        return tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def train(self, loss, global_step):
        """
        Returns train op
        """
        return self.optimizer.minimize(loss, global_step=global_step)

    def print_parameters(self):
        print "\nParameters:"
        print("----------")
        total_parameters = 0
        for v in tf.trainable_variables():
            num_parameters = v.get_shape().num_elements()
            print("{}: {:,}".format(v.name, num_parameters))
            total_parameters += num_parameters
        print("Total Parameters: {:,}\n".format(total_parameters))

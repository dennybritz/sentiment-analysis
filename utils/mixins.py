import tensorflow as tf
import os
import time
import datetime


class NNMixin:

    def _build_input_batches(self, train_x, train_y, num_epochs, batch_size):
        """
        Builds a graph that stores the input in memory and uses queues
        to slice it into bactches.

        Returns a node representing batches of x and y.
        """
        # Use Tensorflow's queues and batching features
        x_slice, y_slice = tf.train.slice_input_producer([train_x, train_y], num_epochs=num_epochs)
        x_batch, y_batch = tf.train.batch([x_slice, y_slice], batch_size=batch_size)
        return [x_batch, y_batch]

    def _build_embedding(self, shape, input_tensor):
        """
        Builds an embedding layer.

        Returns the final embedding.
        """
        # We force this on the CPU because the op isn't implemented for the GPU yet
        with tf.variable_scope("embedding"), tf.device('/cpu:0'):
            W_intializer = tf.random_uniform(shape, -1.0, 1.0)
            W_embeddings = tf.Variable(W_intializer, name="W")
            return tf.nn.embedding_lookup(W_embeddings, input_tensor)

    def _build_affine(self, shape, input_tensor, activation_func=tf.nn.relu):
        """
        Builds an affine (fully-connected) layer
        """
        with tf.variable_scope("affine"):
            W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name="b")
            h = activation_func(tf.matmul(input_tensor, W) + b, name="h")
        return h

    def _build_softmax(self, shape, input_tensor):
        """
        Builds a softmax layer
        """
        W_initializer = tf.truncated_normal_initializer(stddev=0.1)
        b_initializer = tf.constant_initializer(0.1)
        W = tf.get_variable("W", shape, initializer=W_initializer)
        b = tf.get_variable("b", shape[-1:], initializer=b_initializer)
        return tf.nn.softmax(tf.matmul(input_tensor, W) + b, name="y")

    def _build_mean_ce_loss(self, predictions, labels):
        """
        Calculates the mean cross-entropy loss
        """
        with tf.variable_scope("mean-ce-loss"):
            return -tf.reduce_mean(labels * tf.log(predictions), name="mean_ce_loss")

    def _build_total_ce_loss(self, predictions, labels):
        """
        Calculates the mean cross-entropy loss
        """
        with tf.variable_scope("total-ce-loss"):
            return -tf.reduce_sum(labels * tf.log(predictions), name="total_ce_loss")

    def _build_accuracy(self, predictions, labels):
        """
        Returns accuracy tensor
        """
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def print_parameters(self):
        print "\nParameters:"
        print("----------")
        total_parameters = 0
        for v in tf.trainable_variables():
            num_parameters = v.get_shape().num_elements()
            print("{}: {:,}".format(v.name, num_parameters))
            total_parameters += num_parameters
        print("Total Parameters: {:,}\n".format(total_parameters))


class TrainMixin:

    def print_summaries(self, summaries):
        """
        Prints Event summary protocol buffers
        """
        summary_obj = tf.Summary.FromString(summaries)
        # Don't include summaries about queues
        filtered_summaries = [v for v in summary_obj.value if "queue/" not in v.tag]
        summary_str = "\n".join(["{}: {:f}".format(v.tag, v.simple_value) for v in filtered_summaries])
        print("\n{}\n".format(summary_str))

    def build_eval_step(self, out_dir, predict_op, global_step, summary_op, prefix="dev", sess=None):
        """
        Builds a step function that evaluates a given input and prints out summaries.
        Also writes summaries to an event file for Tensorboard.

        Returns a step function that can be called with a feed_dict.
        """
        summary_dir = os.path.abspath(os.path.join(out_dir, "summaries", "dev"))
        eval_writer = tf.train.SummaryWriter(self.train_summary_dir, sess.graph_def)

        # A single evaluation step
        def step(feed_dict=None):
            out_, global_step_, summaries_ = sess.run([predict_op, global_step, summary_op], feed_dict=feed_dict)
            eval_writer.add_summary(summaries_, global_step_)
            self.print_summaries(summaries_)
            return [out_, summaries_, global_step_]

        return step

    def build_train_step(self, out_dir, train_op, global_step, summary_op, ops=[], save_every=16, sess=None):
        """
        Builds a training step function. Also saves summaries and optionally checkpoints model.
        Returns a step function that can be called with a feed_dict.
        """
        if not sess:
            sess = tf.get_default_session()

        # Train writer
        self.train_summary_dir = os.path.abspath(os.path.join(out_dir, "summaries", "train"))
        self.train_writer = tf.train.SummaryWriter(self.train_summary_dir, sess.graph_def)

        # Checkpointing
        self.checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver(tf.all_variables())

        # A single training step
        def step(feed_dict=None):
            # Execute train step
            all_ops = [train_op, global_step, summary_op] + ops
            results = sess.run(all_ops, feed_dict=feed_dict)
            _, global_step_, summaries_ = results[:3]
            other_resuts = results[3:]
            # Print Step
            time_str = datetime.datetime.now().isoformat()
            print("{}: Step {}".format(time_str, global_step_))
            # Write summaries
            self.train_writer.add_summary(summaries_, global_step_)
            # Maybe checkpoint
            if global_step_ % save_every == 0:
                save_path = self.saver.save(sess, self.checkpoint_prefix, global_step_)
                print("\nSaved model parameters to {}\n".format(save_path))
            return [global_step_, summaries_] + other_resuts

        # Return step function
        return step

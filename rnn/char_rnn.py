import tensorflow as tf
from utils.mixins import NNMixin, TrainMixin
from tensorflow.models.rnn import rnn_cell


class CharRNN(object, NNMixin, TrainMixin):
    def __init__(
      self, vocabulary_size, sequence_length, batch_size, num_classes,
      embedding_size=128, hidden_dim=256, cell=None, num_layers=3, loss="linear_gain"):

        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length])
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes])

        if not cell:
            # Standard cell: Stacked LSTM
            first_cell = rnn_cell.LSTMCell(hidden_dim, embedding_size, use_peepholes=True)
            next_cell = rnn_cell.LSTMCell(hidden_dim, hidden_dim, use_peepholes=True)
            self.cell = rnn_cell.MultiRNNCell([first_cell] + [next_cell] * (num_layers - 1))

        with tf.variable_scope("embedding"):
            self.embedded_chars = self._build_embedding([vocabulary_size, embedding_size], self.input_x)

        with tf.variable_scope("rnn") as scope:
            self.state = tf.Variable(tf.zeros([batch_size, self.cell.state_size]))
            self.outputs = []
            self.states = [self.state]
            for i in range(sequence_length):
                if i > 0:
                    scope.reuse_variables()
                new_output, new_state = self.cell(self.embedded_chars[:, i, :], self.states[-1])
                self.outputs.append(new_output)
                self.states.append(new_state)

            self.final_state = self.states[-1]
            self.final_output = self.outputs[-1]

        with tf.variable_scope("softmax"):
            self.ys = [self._build_softmax([hidden_dim, num_classes], o) for o in self.outputs]
            self.y = self.ys[-1]
            self.predictions = tf.argmax(self.y, 1)

        with tf.variable_scope("loss"):
            if loss == "linear_gain":
                # Loss with linear gain. We output at each time step and multiply losses with a linspace
                # Because we have more gradients this can result in faster learning
                packed_ys = tf.pack(self.ys)
                tiled_labels = tf.pack([self.input_y for i in range(sequence_length)])
                accumulated_losses = -tf.reduce_sum(tiled_labels * tf.log(packed_ys), [1, 2])
                loss_gains = tf.linspace(0.0, 1.0, sequence_length)
                annealed_losses = tf.mul(loss_gains, tf.concat(0, accumulated_losses))
                accumulated_loss = tf.reduce_sum(annealed_losses)
                self.loss = accumulated_loss
                self.mean_loss = tf.reduce_mean(annealed_losses)
            elif loss == "last":
                # Standard loss, only last output is considered
                self.loss = self._build_total_ce_loss(self.ys[-1], self.input_y)
                self._build_mean_ce_loss(self.ys[-1], self.input_y)

        # Summaries
        total_loss_summary = tf.scalar_summary("total loss", self.loss)
        mean_loss_summary = tf.scalar_summary("mean loss", self.mean_loss)
        accuracy_summmary = tf.scalar_summary("accuracy", self._build_accuracy(self.y, self.input_y))
        self.summaries = tf.merge_all_summaries()

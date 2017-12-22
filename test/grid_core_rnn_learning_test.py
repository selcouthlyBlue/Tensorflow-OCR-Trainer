import tensorflow as tf
from tensorflow.contrib import grid_rnn


def reshape_to_rnn_dims(tensor, num_time_steps):
    return tf.unstack(tensor, num_time_steps, 1)


class GridLSTMCellTest(tf.test.TestCase):
    def setUp(self):
        self.num_features = 1
        self.time_steps = 1
        self.batch_size = 1
        tf.reset_default_graph()
        self.input_layer = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.num_features])
        self.cell = grid_rnn.Grid1LSTMCell(num_units=8)

    def test_simple_grid_rnn(self):
        self.input_layer = reshape_to_rnn_dims(self.input_layer, self.time_steps)
        tf.nn.static_rnn(self.cell, self.input_layer, dtype=tf.float32)

    def test_dynamic_grid_rnn(self):
        tf.nn.dynamic_rnn(self.cell, self.input_layer, dtype=tf.float32)


class BidirectionalGridRNNCellTest(tf.test.TestCase):
    def setUp(self):
        self.num_features = 1
        self.time_steps = 1
        self.batch_size = 1
        tf.reset_default_graph()
        self.input_layer = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.num_features])
        self.cell_fw = grid_rnn.Grid1LSTMCell(num_units=8)
        self.cell_bw = grid_rnn.Grid1LSTMCell(num_units=8)

    def test_simple_bidirectional_grid_rnn(self):
        self.input_layer = reshape_to_rnn_dims(self.input_layer, self.time_steps)
        tf.nn.static_bidirectional_rnn(self.cell_fw, self.cell_fw, self.input_layer, dtype=tf.float32)

    def test_bidirectional_dynamic_grid_rnn(self):
        tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.input_layer, dtype=tf.float32)


if __name__ == '__main__':
    tf.test.main()

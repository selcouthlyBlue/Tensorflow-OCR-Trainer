import tensorflow as tf
from tensorflow.contrib.ndlstm.python import lstm2d


class MultidimensionalRNNTest(tf.test.TestCase):
    def setUp(self):
        self.num_features = 32
        self.time_steps = 64
        self.batch_size = 1
        self.num_channels = 1
        self.input_layer = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.num_features, self.num_channels])

    def test_simple_mdrnn(self):
        mdlstm = lstm2d.separable_lstm(self.input_layer, 16)
        print(mdlstm)


if __name__ == '__main__':
    tf.test.main()

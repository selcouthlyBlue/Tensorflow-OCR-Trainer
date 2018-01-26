import tensorflow as tf
import tfutils as network_utils
import numpy as np


class MultidimensionalRNNTest(tf.test.TestCase):
    def setUp(self):
        self.num_classes = 26
        self.num_features = 32
        self.time_steps = 1596
        self.batch_size = 3
        self.num_channels = 1
        self.input_layer = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.num_features, self.num_channels])
        self.labels = tf.sparse_placeholder(tf.int32)

    def test_simple_mdrnn(self):
        net = network_utils.mdlstm(self.input_layer, 16)

    def test_image_to_sequence(self):
        net = network_utils.mdlstm(self.input_layer, 16)
        net = network_utils.images_to_sequence(net)

    def test_convert_to_ctc_dims(self):
        net = network_utils.mdlstm(self.input_layer, 16)
        net = network_utils.images_to_sequence(net)

        seq_lens = np.full(self.batch_size, 32, dtype=int)

        net = network_utils.get_time_major(net, batch_size=self.batch_size, num_hidden_units=16, num_classes=self.num_classes)

        loss = network_utils.ctc_loss(labels=self.labels, inputs=net, sequence_length=seq_lens)

    def test_stack_ndlstms(self):
        net = network_utils.mdlstm(self.input_layer, 16)
        net = network_utils.mdlstm(net, 16)


if __name__ == '__main__':
    tf.test.main()

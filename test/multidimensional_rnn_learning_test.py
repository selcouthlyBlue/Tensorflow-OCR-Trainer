import tensorflow as tf
import tfutils as network_utils


class MultidimensionalRNNTest(tf.test.TestCase):
    def setUp(self):
        self.num_classes = 26
        self.num_features = 32
        self.time_steps = 1596
        self.batch_size = 3
        self.num_channels = 1
        self.input_layer = tf.placeholder(tf.float32, [self.batch_size, self.num_features, self.time_steps, self.num_channels])
        self.labels = tf.sparse_placeholder(tf.int32)

    def test_simple_mdrnn(self):
        net = network_utils.mdrnn(self.input_layer, 16)

    def test_image_to_sequence(self):
        net = network_utils.mdrnn(self.input_layer, 16)
        net = network_utils.images_to_sequence(net)

    def test_stack_ndlstms(self):
        net = network_utils.mdrnn(self.input_layer, 16)
        net = network_utils.mdrnn(net, 16)


if __name__ == '__main__':
    tf.test.main()

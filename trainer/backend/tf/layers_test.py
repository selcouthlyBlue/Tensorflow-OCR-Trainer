import tensorflow as tf

from trainer.backend.tf.layers import images_to_sequence
from trainer.backend.tf.layers import mdrnn
from trainer.backend.tf.layers import sequence_to_images


def _create_input(shape):
    return tf.placeholder(tf.float32, shape)


class MDRNNTest(tf.test.TestCase):
    def testImagesToSequenceDims(self):
        inputs = _create_input([2, 7, 11, 5])
        outputs = images_to_sequence(inputs)
        expected_shape = (14, 11, 5)
        self._testAssertShapesAreEqual(outputs, expected_shape)

    def _testAssertShapesAreEqual(self, result, expected_shape):
        self.assertEqual(tuple(result.get_shape().as_list()), expected_shape)

    def testImagesToSequenceDimsDynamicBatchSize(self):
        inputs = tf.placeholder(tf.float32, [None, 7, 11, 5])
        outputs = images_to_sequence(inputs)
        expected_shape = (None, 11, 5)
        self._testAssertShapesAreEqual(outputs, expected_shape)

    def testSequenceToImagesDims(self):
        inputs = _create_input([14, 11, 5])
        outputs = sequence_to_images(inputs, 7)
        expected_shape = (2, 7, 11, 5)
        self._testAssertShapesAreEqual(outputs, expected_shape)

    def testSequenceToImagesDimsDynamicBatchSize(self):
        inputs = _create_input([None, 11, 5])
        outputs = sequence_to_images(inputs, 7)
        expected_shape = (None, 7, 11, 5)
        self._testAssertShapesAreEqual(outputs, expected_shape)

    def testMDRNN(self):
        inputs = _create_input([2, 7, 11, 5])
        outputs = mdrnn(inputs, num_hidden=8)
        expected_shape = (2, 7, 11, 8)
        self._testAssertShapesAreEqual(outputs, expected_shape)

    def testMRNNDynamicBatchSize(self):
        inputs = _create_input([None, 7, 11, 5])
        outputs = mdrnn(inputs, num_hidden=8)
        expected_shape = (None, 7, 11, 8)
        self._testAssertShapesAreEqual(outputs, expected_shape)

    def testMDRNNUsingBlocks(self):
        inputs = _create_input([2, 7, 11, 5])
        outputs = mdrnn(inputs, num_hidden=8, kernel_size=2)
        expected_shape = (2, 4, 6, 8)
        self._testAssertShapesAreEqual(outputs, expected_shape)

    def testMRNNUsingBlocksDynamicBatchSize(self):
        inputs = _create_input([None, 7, 11, 5])
        outputs = mdrnn(inputs, num_hidden=8, kernel_size=2)
        expected_shape = (None, 4, 6, 8)
        self._testAssertShapesAreEqual(outputs, expected_shape)


if __name__ == "__main__":
    tf.test.main()

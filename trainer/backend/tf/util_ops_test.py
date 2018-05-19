import tensorflow as tf

from trainer.backend.tf.layers import bidirectional_rnn
from trainer.backend.tf.ctc_ops import convert_to_ctc_dims, ctc_beam_search_decoder
from trainer.backend.tf.util_ops import get_sequence_lengths
from trainer.backend.tf.losses import ctc_loss


def _create_input(shape):
    return tf.placeholder(tf.float32, shape)


class GetSequenceLengthTest(tf.test.TestCase):
    def setUp(self):
        self.inputs = _create_input([1, 128, 64])
        self.inputs = bidirectional_rnn(self.inputs, 8)
        self.labels = tf.sparse_placeholder(dtype=tf.int32)
        self.sequence_lengths = get_sequence_lengths(self.inputs)

    def testCTCLOSS(self):
        ctc_inputs = convert_to_ctc_dims(self.inputs,
                                         num_classes=79,
                                         num_steps=self.inputs.shape[1],
                                         num_outputs=self.inputs.shape[-1])
        ctc_loss(self.labels, ctc_inputs, self.sequence_lengths)

    def testCTCDecoder(self):
        ctc_inputs = convert_to_ctc_dims(self.inputs,
                                         num_classes=79,
                                         num_steps=self.inputs.shape[1],
                                         num_outputs=self.inputs.shape[-1])
        ctc_beam_search_decoder(ctc_inputs, self.sequence_lengths)


if __name__ == "__main__":
    tf.test.main()

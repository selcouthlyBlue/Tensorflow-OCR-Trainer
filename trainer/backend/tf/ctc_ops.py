import tensorflow as tf
from trainer.backend.tf.util_ops import get_sequence_lengths
from tensorflow.contrib import slim


def ctc_beam_search_decoder(inputs, merge_repeated=True):
    sequence_length = get_sequence_lengths(inputs)
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs, sequence_length, merge_repeated)
    return decoded[0], log_probabilities


def convert_to_ctc_dims(inputs, num_classes, num_steps, num_outputs):
    outputs = tf.reshape(inputs, [-1, num_outputs])
    logits = slim.fully_connected(outputs, num_classes)
    logits = tf.reshape(logits, [-1, num_steps, num_classes])
    logits = tf.transpose(logits, (1, 0, 2))
    return logits

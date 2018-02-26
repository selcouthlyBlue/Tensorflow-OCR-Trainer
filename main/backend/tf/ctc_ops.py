import tensorflow as tf

from backend.tf.util_ops import get_sequence_lengths


def ctc_beam_search_decoder(inputs, merge_repeated=True):
    sequence_length = get_sequence_lengths(inputs)
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs, sequence_length, merge_repeated)
    return decoded[0], log_probabilities

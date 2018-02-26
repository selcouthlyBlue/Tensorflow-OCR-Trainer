import tensorflow as tf

from backend.tf.util_ops import get_sequence_lengths, dense_to_sparse


def ctc_loss(labels, inputs, preprocess_collapse_repeated_labels=True,
             ctc_merge_repeated=True, inputs_are_time_major=True,
             eos_token=0):
    sequence_length = get_sequence_lengths(inputs)
    sparse_labels = dense_to_sparse(labels, eos_token=eos_token)
    return tf.reduce_mean(tf.nn.ctc_loss(sparse_labels, inputs, sequence_length,
                          preprocess_collapse_repeated=preprocess_collapse_repeated_labels,
                          ctc_merge_repeated=ctc_merge_repeated,
                          time_major=inputs_are_time_major))

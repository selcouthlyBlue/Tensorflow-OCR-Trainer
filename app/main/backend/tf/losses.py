import tensorflow as tf

def ctc_loss(labels, inputs, sequence_length, preprocess_collapse_repeated_labels=False,
             ctc_merge_repeated=True, inputs_are_time_major=True):
    return tf.reduce_mean(tf.nn.ctc_loss(labels, inputs, sequence_length,
                          preprocess_collapse_repeated=preprocess_collapse_repeated_labels,
                          ctc_merge_repeated=ctc_merge_repeated,
                          time_major=inputs_are_time_major))

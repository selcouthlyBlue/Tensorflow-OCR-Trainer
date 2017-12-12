import tensorflow as tf
import tflearn
from tflearn import bidirectional_rnn, BasicLSTMCell

from optimizer_enum import Optimizers

def ctc_loss(predictions, labels: tf.SparseTensor, sequence_length,
             preprocess_collapse_repeated_labels=True,
             ctc_merge_repeated=True,
             inputs_are_time_major=True):
    return tf.nn.ctc_loss(predictions, labels, sequence_length,
                          preprocess_collapse_repeated_labels,
                          ctc_merge_repeated,
                          inputs_are_time_major)

def input_data(shape, name: str = 'InputData', input_type=tf.float32):
    return tflearn.input_data(shape=shape, dtype=input_type, name=name)

def reshape(tensor: tf.Tensor, new_shape: list):
    return tf.reshape(tensor, new_shape, name="reshape")

def bidirectional_lstm(inputs, num_hidden: int, return_seq=False):
    return bidirectional_rnn(inputs, BasicLSTMCell(num_hidden), BasicLSTMCell(num_hidden), return_seq=return_seq)


def decode(inputs, sequence_length, merge_repeated=True):
    decoded, _ = tf.nn.ctc_beam_search_decoder(inputs, sequence_length, merge_repeated)
    decoded = tf.to_int32(decoded)
    return tf.sparse_to_dense(sparse_indices=decoded[0].indices,
                              output_shape=decoded[0].dense_shape,
                              sparse_values=decoded[0].values,
                              name="output")

def label_error_rate(y_pred, y_true):
    return tf.reduce_mean(tf.edit_distance(tf.cast(y_pred, tf.int32), y_true))

def optimize(loss, optimizer, learning_rate):
    if optimizer == Optimizers.ADAM:
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    if optimizer == Optimizers.ADADELTA:
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    if optimizer == Optimizers.RMSPROP:
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    raise NotImplementedError("{} is not implemented.".format(optimizer))

def label_data(name, input_type=tf.int32):
    return tf.sparse_placeholder(input_type, name=name)

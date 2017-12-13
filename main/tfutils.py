import tensorflow as tf
import tflearn
from tflearn import bidirectional_rnn, BasicLSTMCell

from optimizer_enum import Optimizers

def ctc_loss(predictions, labels, sequence_length,
             preprocess_collapse_repeated_labels=True,
             ctc_merge_repeated=True,
             inputs_are_time_major=True):
    return tf.nn.ctc_loss(inputs=predictions, labels=labels, sequence_length=sequence_length,
                          preprocess_collapse_repeated=preprocess_collapse_repeated_labels,
                          ctc_merge_repeated=ctc_merge_repeated,
                          time_major=inputs_are_time_major)

def input_data(shape, name: str = 'InputData', input_type=tf.float32):
    return tflearn.input_data(shape=shape, dtype=input_type, name=name)

def reshape(tensor: tf.Tensor, new_shape: list):
    return tf.reshape(tensor, new_shape, name="reshape")

def bidirectional_lstm(inputs, num_hidden: int, return_seq=False):
    return bidirectional_rnn(inputs, BasicLSTMCell(num_hidden), BasicLSTMCell(num_hidden), return_seq=return_seq)


def decode(inputs, sequence_length, merge_repeated=True):
    decoded, _ = tf.nn.ctc_beam_search_decoder(inputs, sequence_length, merge_repeated)
    return decoded

def label_error_rate(y_pred, y_true):
    return tf.reduce_mean(tf.edit_distance(tf.cast(y_pred, tf.int32), y_true))

def optimize(loss, optimizer, learning_rate):
    if optimizer == Optimizers.MOMENTUM:
        return tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss)
    if optimizer == Optimizers.ADAM:
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    if optimizer == Optimizers.ADADELTA:
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    if optimizer == Optimizers.RMSPROP:
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    raise NotImplementedError("{} is not implemented.".format(optimizer))

def sparse_input_data(input_type=tf.int32):
    return tf.sparse_placeholder(input_type)

def get_time_major(model, num_classes, batch_size, num_hidden_units):
    outputs = reshape(model, [-1, num_hidden_units])

    W = tf.Variable(tf.truncated_normal([num_hidden_units,
                                         num_classes],
                                        stddev=0.1, dtype=tf.float32), name='W')
    b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_size, -1, num_classes])
    logits = tf.transpose(logits, (1, 0, 2))
    return logits

def cost(loss):
    return tf.reduce_mean(loss)


def get_type(type_str):
    if type_str == 'int32':
        return tf.int32
    return tf.float32


def get_shape(tensor):
    return tf.shape(tensor)
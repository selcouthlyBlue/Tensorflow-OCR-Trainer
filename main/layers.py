import tensorflow as tf

from tensorflow.contrib import rnn, slim

def reshape(tensor: tf.Tensor, new_shape: list, name="reshape"):
    return tf.reshape(tensor, new_shape, name=name)


def bidirectional_rnn(inputs, num_hidden, cell_type='LSTM', concat_output=True):
    cell_fw = _get_cell(num_hidden, cell_type)
    cell_bw = _get_cell(num_hidden, cell_type)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                 cell_bw,
                                                 inputs,
                                                 dtype=tf.float32)
    if concat_output:
        return tf.concat(outputs, 2)
    return outputs


def _get_cell(num_filters_out, cell_type='LSTM'):
    if cell_type == 'LSTM':
        return rnn.LSTMCell(num_filters_out, initializer=slim.xavier_initializer())
    if cell_type == 'GRU':
        return rnn.GRUCell(num_filters_out, kernel_initializer=slim.xavier_initializer())
    if cell_type == 'GLSTM':
        return rnn.GLSTMCell(num_filters_out, initializer=slim.xavier_initializer())
    raise NotImplementedError(cell_type, "is not supported.")


def mdrnn(inputs, num_hidden, cell_type='LSTM', scope=None):
    with tf.variable_scope(scope, "multidimensional_rnn", [inputs]):
        hidden_sequence_horizontal = _bidirectional_rnn_scan(inputs, num_hidden // 2, cell_type=cell_type)
        with tf.variable_scope("vertical"):
            transposed = tf.transpose(hidden_sequence_horizontal, [0, 2, 1, 3])
            output_transposed = _bidirectional_rnn_scan(transposed, num_hidden // 2, cell_type=cell_type)
        output = tf.transpose(output_transposed, [0, 2, 1, 3])
        return output


def images_to_sequence(inputs):
    _, _, width, num_channels = inputs.get_shape().as_list()
    s = tf.shape(inputs)
    batch_size, height = s[0], s[1]
    transposed = tf.transpose(inputs, [2, 0, 1, 3])
    return reshape(transposed, [width, batch_size * height, num_channels])


def sequence_to_images(tensor, height):
    width, num_batches, depth = tensor.get_shape().as_list()
    if num_batches is None:
        num_batches = -1
    else:
        num_batches = num_batches // height
    reshaped = tf.reshape(tensor,
                                 [width, num_batches, height, depth])
    return tf.transpose(reshaped, [1, 2, 0, 3])


def _bidirectional_rnn_scan(inputs, num_hidden, cell_type='LSTM'):
    with tf.variable_scope("BidirectionalRNN", [inputs]):
        height = inputs.get_shape().as_list()[1]
        inputs = images_to_sequence(inputs)
        output_sequence = bidirectional_rnn(inputs, num_hidden, cell_type)
        output = sequence_to_images(output_sequence, height)
        return output


def conv2d(inputs, num_filters, kernel, activation_fn=tf.nn.relu, scope=None):
    return slim.conv2d(inputs, num_filters, kernel,
                       scope=scope, activation_fn=activation_fn)


def max_pool2d(inputs, kernel, scope=None):
    return slim.max_pool2d(inputs, kernel, scope=scope)


def convert_to_ctc_dims(inputs, num_classes, num_steps, num_hidden_units):
    outputs = reshape(inputs, [-1, num_hidden_units])
    logits = slim.fully_connected(outputs, num_classes,
                                  weights_initializer=slim.xavier_initializer())
    logits = reshape(logits, [num_steps, -1, num_classes])
    return logits


def dropout(inputs, keep_prob, is_training, scope=None):
    return slim.dropout(inputs, keep_prob, scope=scope, is_training=is_training)


def collapse_to_rnn_dims(inputs):
    batch_size, height, width, num_channels = inputs.get_shape().as_list()
    if batch_size is None:
        batch_size = -1
    return tf.reshape(inputs, [batch_size, width, height * num_channels])


def batch_norm(inputs, is_training):
    return slim.batch_norm(inputs, is_training=is_training)


def l2_normalize(inputs, axis):
    return tf.nn.l2_normalize(inputs, axis)

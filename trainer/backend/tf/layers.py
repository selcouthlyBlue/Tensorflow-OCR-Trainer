import tensorflow as tf

from tensorflow.contrib import rnn, slim


def get_sequence_lengths(inputs):
    used = tf.sign(tf.reduce_max(tf.abs(inputs), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def reshape(tensor: tf.Tensor, new_shape: list, name="reshape"):
    return tf.reshape(tensor, new_shape, name=name)


def bidirectional_rnn(inputs, num_hidden, cell_type='LSTM',
                      activation='tanh', concat_output=True,
                      scope=None):
    with tf.variable_scope(scope, "bidirectional_rnn", [inputs]):
        cell_fw = _get_cell(num_hidden, cell_type, activation)
        cell_bw = _get_cell(num_hidden, cell_type, activation)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                     cell_bw,
                                                     inputs,
                                                     sequence_length=get_sequence_lengths(inputs),
                                                     dtype=tf.float32)
        if concat_output:
            return tf.concat(outputs, 2)
        return outputs


def _get_activation(name):
    if name == 'tanh':
        return tf.nn.tanh
    if name == 'relu':
        return tf.nn.relu
    if name == 'relu6':
        return tf.nn.relu6
    raise NotImplementedError(name, "activation function not implemented")


def _get_cell(num_filters_out, cell_type='LSTM', activation='tanh'):
    cell_type = cell_type or 'LSTM'
    activation = activation or 'tanh'
    activation_function = _get_activation(activation)
    if cell_type == 'LSTM':
        return rnn.LSTMCell(num_filters_out,
                            initializer=slim.xavier_initializer(),
                            activation=activation_function)
    if cell_type == 'GRU':
        return rnn.GRUCell(num_filters_out,
                           kernel_initializer=slim.xavier_initializer(),
                           activation=activation_function)
    if cell_type == 'GLSTM':
        return rnn.GLSTMCell(num_filters_out,
                             initializer=slim.xavier_initializer(),
                             activation=activation_function)
    raise NotImplementedError(cell_type, "is not supported.")


def mdrnn(inputs, num_hidden, cell_type='LSTM', activation='tanh', scope=None):
    with tf.variable_scope(scope, "multidimensional_rnn", [inputs]):
        hidden_sequence_horizontal = _bidirectional_rnn_scan(inputs,
                                                             num_hidden // 2,
                                                             cell_type=cell_type,
                                                             activation=activation)
        with tf.variable_scope("vertical"):
            transposed = tf.transpose(hidden_sequence_horizontal, [0, 2, 1, 3])
            output_transposed = _bidirectional_rnn_scan(transposed, num_hidden // 2, cell_type=cell_type)
        output = tf.transpose(output_transposed, [0, 2, 1, 3])
        return output


def images_to_sequence(inputs):
    _, _, width, num_channels = _get_shape_as_list(inputs)
    s = tf.shape(inputs)
    batch_size, height = s[0], s[1]
    return reshape(inputs, [batch_size * height, width, num_channels])


def _get_shape_as_list(tensor):
    return tensor.get_shape().as_list()


def sequence_to_images(tensor, height):
    num_batches, width, depth = tensor.get_shape().as_list()
    if num_batches is None:
        num_batches = -1
    else:
        num_batches = num_batches // height
    reshaped = tf.reshape(tensor,
                          [num_batches, width, height, depth])
    return tf.transpose(reshaped, [0, 2, 1, 3])


def _bidirectional_rnn_scan(inputs, num_hidden, cell_type='LSTM', activation='tanh'):
    with tf.variable_scope("BidirectionalRNN", [inputs]):
        height = inputs.get_shape().as_list()[1]
        inputs = images_to_sequence(inputs)
        output_sequence = bidirectional_rnn(inputs, num_hidden, cell_type, activation)
        output = sequence_to_images(output_sequence, height)
        return output


def conv2d(inputs, num_filters, kernel_size, activation="relu", stride=1, padding='SAME', scope=None):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    padding = padding or 'SAME'
    activation = activation or "relu"
    return slim.conv2d(inputs, num_filters, kernel_size,
                       activation_fn=_get_activation(activation),
                       padding=padding,
                       stride=stride,
                       scope=scope)


def max_pool2d(inputs, pool_size, padding='VALID', stride=2, scope=None):
    if isinstance(pool_size, int):
        pool_size = [pool_size, pool_size]
    padding = padding or 'VALID'
    stride = stride or 2
    return slim.max_pool2d(inputs, pool_size, padding=padding, stride=stride, scope=scope)


def dropout(inputs, keep_prob, is_training, scope=None):
    return slim.dropout(inputs, keep_prob, scope=scope, is_training=is_training)


def collapse_to_rnn_dims(inputs):
    batch_size, height, width, num_channels = inputs.get_shape().as_list()
    if batch_size is None:
        batch_size = -1
    nwhc_cnn_outputs = tf.transpose(inputs, (0, 2, 1, 3))
    batch_major_rnn_inputs = tf.reshape(nwhc_cnn_outputs,
                                        [batch_size, width, height * num_channels]
                                        )
    return batch_major_rnn_inputs


def batch_norm(inputs, is_training):
    return slim.batch_norm(inputs, is_training=is_training)


def l2_normalize(inputs, axis):
    return tf.nn.l2_normalize(inputs, axis)

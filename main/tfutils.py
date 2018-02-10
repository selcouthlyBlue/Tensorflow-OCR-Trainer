import tensorflow as tf

from optimizer_enum import Optimizers
from tensorflow.contrib import grid_rnn, learn, rnn
from six.moves import xrange
from tensorflow.contrib import slim


def ctc_loss(labels, inputs, sequence_length,
             preprocess_collapse_repeated_labels=True,
             ctc_merge_repeated=True,
             inputs_are_time_major=True):
    return tf.reduce_mean(tf.nn.ctc_loss(labels, inputs, sequence_length,
                          preprocess_collapse_repeated=preprocess_collapse_repeated_labels,
                          ctc_merge_repeated=ctc_merge_repeated,
                          time_major=inputs_are_time_major))


def reshape(tensor: tf.Tensor, new_shape: list):
    return tf.reshape(tensor, new_shape, name="reshape")


def bidirectional_grid_lstm(inputs, num_hidden):
    cell_fw = grid_rnn.Grid2LSTMCell(num_units=num_hidden)
    cell_bw = grid_rnn.Grid2LSTMCell(num_units=num_hidden)
    return tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32)[0]


def _get_cell(num_filters_out, cell_type='LSTM'):
    if cell_type == 'LSTM':
        return rnn.BasicLSTMCell(num_filters_out)
    if cell_type == 'Grid1LSTM':
        return grid_rnn.Grid1LSTMCell(num_filters_out)
    if cell_type == 'Grid2LSTM':
        return grid_rnn.Grid2LSTMCell(num_filters_out, tied=True)
    if cell_type == 'Grid3LSTM':
        return grid_rnn.Grid3LSTMCell(num_filters_out, tied=True)
    raise NotImplementedError(cell_type, "is not supported.")


def mdlstm(inputs, num_filters_out, cell_type='LSTM', scope=None):
    with tf.variable_scope(scope, "multidimensional_rnn", [inputs]):
        cells = [_get_cell(num_filters_out // 2, cell_type) for _ in xrange(4)]
        cell_fw1 = cells[0]
        cell_bw1 = cells[1]
        cell_fw2 = cells[2]
        cell_bw2 = cells[3]
        hidden_sequence_horizontal = _bidirectional_rnn_scan(cell_fw1, cell_bw1, inputs)
        with tf.variable_scope("vertical"):
            transposed = tf.transpose(hidden_sequence_horizontal, [0, 2, 1, 3])
            output_transposed = _bidirectional_rnn_scan(cell_fw2, cell_bw2, transposed)
        output = tf.transpose(output_transposed, [0, 2, 1, 3])
        return output


def _scan(cell, inputs, reverse=False):
    with tf.variable_scope("SeqLSTM", [inputs, cell]):
        if reverse:
            inputs = tf.reverse_v2(inputs, [0])
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
        if not isinstance(outputs, tf.Tensor):
            outputs = outputs[0]
        if reverse:
            outputs = tf.reverse_v2(outputs, [0])
        return outputs


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


def _bidirectional_rnn_scan(cell_fw, cell_bw, inputs):
    with tf.variable_scope("BidirectionalRNN", [inputs, cell_fw, cell_bw]):
        height = inputs.get_shape().as_list()[1]
        inputs = images_to_sequence(inputs)
        with tf.variable_scope("lr"):
            hidden_sequence_lr = _scan(cell_fw, inputs)
        with tf.variable_scope("rl"):
            hidden_sequence_rl = _scan(cell_bw, inputs, reverse=True)
        output_sequence = tf.concat([hidden_sequence_lr, hidden_sequence_rl], 2)
        output = sequence_to_images(output_sequence, height)
        return output


def conv2d(inputs, num_filters_out, kernel, activation_fn=tf.nn.tanh, scope=None):
    return slim.conv2d(inputs, num_filters_out, kernel, scope=scope, activation_fn=activation_fn)


def max_pool2d(inputs, kernel, scope=None):
    return slim.max_pool2d(inputs, kernel, scope=scope)


def ctc_beam_search_decoder(inputs, sequence_length, merge_repeated=True):
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs, sequence_length, merge_repeated)
    return decoded[0], log_probabilities


def sparse_to_dense(sparse_tensor, name="sparse_to_dense"):
    return tf.sparse_to_dense(tf.to_int32(sparse_tensor.indices),
                              tf.to_int32(sparse_tensor.values),
                              tf.to_int32(sparse_tensor.dense_shape),
                              name=name)


def accuracy(y_pred, y_true):
    return tf.subtract(tf.constant(1, dtype=tf.float32),
                       tf.reduce_mean(tf.edit_distance(tf.cast(y_pred, tf.int32), y_true)),
                       name="accuracy")


def optimize(loss, optimizer_name, learning_rate):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = get_optimizer(learning_rate, optimizer_name)
    return optimizer.minimize(loss, global_step=global_step)


def get_optimizer(learning_rate, optimizer_name):
    if optimizer_name == Optimizers.MOMENTUM:
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif optimizer_name == Optimizers.ADAM:
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == Optimizers.ADADELTA:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    return optimizer


def get_logits(inputs, num_classes, num_steps, num_hidden_units):
    outputs = reshape(inputs, [-1, num_hidden_units])
    logits = slim.fully_connected(outputs, num_classes)
    logits = reshape(logits, [num_steps, -1, num_classes])
    return logits


def get_shape(tensor):
    return tf.shape(tensor)


def dense_to_sparse(tensor, eos_token=0):
    indices = tf.where(tf.not_equal(tensor, tf.constant(eos_token, dtype=tensor.dtype)))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def dropout(inputs, rate, scope=None):
    return slim.dropout(inputs, rate, scope=scope)


def div(inputs, divisor, is_floor=True):
    if is_floor:
        return tf.to_int32(tf.floor_div(inputs, tf.constant(divisor, dtype=inputs.dtype)))
    return tf.to_int32(tf.ceil(tf.truediv(inputs, tf.constant(divisor, dtype=inputs.dtype))))


def run_experiment(model, train_input_fn, checkpoint_dir, validation_input_fn=None,
                   validation_steps=100):
    estimator = learn.Estimator(model_fn=model.model_fn,
                                params=model.params,
                                model_dir=checkpoint_dir,
                                config=learn.RunConfig(save_checkpoints_steps=validation_steps))
    estimator.fit(input_fn=train_input_fn)
    estimator.evaluate(input_fn=validation_input_fn)


def input_fn(x_feed_dict, y, num_epochs=1, shuffle=True, batch_size=1):
    return tf.estimator.inputs.numpy_input_fn(x=x_feed_dict,
                                              y=y,
                                              shuffle=shuffle,
                                              num_epochs=num_epochs,
                                              batch_size=batch_size)


def collapse_to_rnn_dims(inputs):
    batch_size, height, width, num_channels = inputs.get_shape().as_list()
    if batch_size is None:
        batch_size = -1
    return tf.reshape(inputs, [batch_size, height * width, num_channels])

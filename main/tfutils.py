import tensorflow as tf

from optimizer_enum import Optimizers
from tensorflow.contrib import grid_rnn, learn
from tensorflow.contrib.ndlstm.python import lstm2d
from tensorflow.contrib import slim


def ctc_loss(labels, inputs, sequence_length,
             preprocess_collapse_repeated_labels=True,
             ctc_merge_repeated=True,
             inputs_are_time_major=True):
    return tf.nn.ctc_loss(labels, inputs, sequence_length,
                          preprocess_collapse_repeated=preprocess_collapse_repeated_labels,
                          ctc_merge_repeated=ctc_merge_repeated,
                          time_major=inputs_are_time_major)


def reshape(tensor: tf.Tensor, new_shape: list):
    return tf.reshape(tensor, new_shape, name="reshape")


def bidirectional_grid_lstm(inputs, num_hidden):
    cell_fw = grid_rnn.Grid2LSTMCell(num_units=num_hidden)
    cell_bw = grid_rnn.Grid2LSTMCell(num_units=num_hidden)
    return tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32)[0]


def mdlstm(inputs, num_filters_out, kernel_size=None, nhidden=None, scope=None):
    return lstm2d.separable_lstm(inputs, num_filters_out, kernel_size=kernel_size, nhidden=nhidden, scope=scope)


def conv2d(inputs, num_filters_out, kernel, scope=None):
    return slim.conv2d(inputs, num_filters_out, kernel, scope=scope)


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


def get_time_major(inputs, num_classes, batch_size, num_hidden_units):
    outputs = reshape(inputs, [-1, num_hidden_units])

    W = tf.Variable(tf.truncated_normal([num_hidden_units,
                                         num_classes],
                                        stddev=0.1, dtype=tf.float32), name='W')
    b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_size, -1, num_classes])
    logits = tf.transpose(logits, (1, 0, 2))
    return logits


def get_shape(tensor):
    return tf.shape(tensor)


def cost(loss):
    return tf.reduce_mean(loss)


def dense_to_sparse(tensor, eos_token=0):
    indices = tf.where(tf.not_equal(tensor, tf.constant(eos_token, dtype=tensor.dtype)))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def dropout(inputs, rate, scope=None):
    return slim.dropout(inputs, rate, scope=scope)


def images_to_sequence(inputs):
    return lstm2d.images_to_sequence(inputs)


def run_experiment(model, train_input_fn, checkpoint_dir, num_epochs=None, validation_input_fn=None,
                   validation_steps=100):
    validation_monitor = learn.monitors.ValidationMonitor(input_fn=validation_input_fn, every_n_steps=validation_steps)
    estimator = learn.Estimator(model_fn=model.model_fn, params=model.params, model_dir=checkpoint_dir)
    estimator.fit(input_fn=train_input_fn, steps=num_epochs, monitors=[validation_monitor])


def input_fn(x_feed_dict, y, num_epochs=1, shuffle=True, batch_size=1):
    return tf.estimator.inputs.numpy_input_fn(x=x_feed_dict,
                                              y=y,
                                              shuffle=shuffle,
                                              num_epochs=num_epochs,
                                              batch_size=batch_size)

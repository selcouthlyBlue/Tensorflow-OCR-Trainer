import tensorflow as tf
import json

from tensorflow.contrib import learn, rnn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn import ModeKeys
from six.moves import xrange
from tensorflow.contrib import slim

tf.logging.set_verbosity(tf.logging.INFO)

def ctc_loss(labels, inputs, sequence_length,
             preprocess_collapse_repeated_labels=True,
             ctc_merge_repeated=True,
             inputs_are_time_major=True):
    return tf.reduce_mean(tf.nn.ctc_loss(labels, inputs, sequence_length,
                          preprocess_collapse_repeated=preprocess_collapse_repeated_labels,
                          ctc_merge_repeated=ctc_merge_repeated,
                          time_major=inputs_are_time_major))


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
        cells = [_get_cell(num_hidden // 2, cell_type) for _ in xrange(4)]
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
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=inputs.dtype)
        output_sequence = tf.concat(outputs, 2)
        output = sequence_to_images(output_sequence, height)
        return output


def conv2d(inputs, num_filters, kernel, activation_fn=tf.nn.relu, scope=None):
    return slim.conv2d(inputs, num_filters, kernel,
                       scope=scope, activation_fn=activation_fn)


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


def label_error_rate(y_pred, y_true):
    return tf.reduce_mean(tf.edit_distance(tf.cast(y_pred, tf.int32), y_true), name="label_error_rate")


def get_optimizer(learning_rate, optimizer_name):
    if optimizer_name == "momentum":
        return tf.train.MomentumOptimizer(learning_rate,
                                          momentum=0.9,
                                          use_nesterov=True)
    elif optimizer_name == "adam":
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == "adadelta":
        return tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer_name == "rmsprop":
        return tf.train.RMSPropOptimizer(learning_rate)
    raise NotImplementedError(optimizer_name + " optimizer not supported")


def convert_to_ctc_dims(inputs, num_classes, num_steps, num_hidden_units):
    outputs = reshape(inputs, [-1, num_hidden_units])
    logits = slim.fully_connected(outputs, num_classes,
                                  weights_initializer=slim.xavier_initializer())
    logits = reshape(logits, [num_steps, -1, num_classes])
    return logits


def dense_to_sparse(tensor, eos_token=0):
    indices = tf.where(tf.not_equal(tensor, tf.constant(eos_token, dtype=tensor.dtype)))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def dropout(inputs, keep_prob, mode, scope=None):
    return slim.dropout(inputs, keep_prob, scope=scope, is_training=is_training(mode))


def div(inputs, divisor, is_floor=True):
    if is_floor:
        return tf.to_int32(tf.floor_div(inputs, tf.constant(divisor, dtype=inputs.dtype)))
    return tf.to_int32(tf.ceil(tf.truediv(inputs, tf.constant(divisor, dtype=inputs.dtype))))


def run_experiment(model_config_file, train_input_fn, checkpoint_dir, validation_input_fn=None,
                   validation_steps=100):
    validation_monitor = learn.monitors.ValidationMonitor(input_fn=validation_input_fn,
                                                          every_n_steps=validation_steps)
    estimator = learn.Estimator(model_fn=model_fn,
                                params=json.load(open(model_config_file, 'r')),
                                model_dir=checkpoint_dir,
                                config=learn.RunConfig(save_checkpoints_steps=validation_steps))
    estimator.fit(input_fn=train_input_fn, monitors=[validation_monitor])


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
    return tf.reshape(inputs, [batch_size, width, height * num_channels])


def add_to_summary(name, value):
    tf.summary.scalar(name, value)


def create_train_op(loss, learning_rate, optimizer):
    optimizer = get_optimizer(learning_rate, optimizer)
    return slim.learning.create_train_op(loss, optimizer, global_step=tf.train.get_or_create_global_step())


def create_model_fn(mode, predictions, loss=None, train_op=None, eval_metric_ops=None):
    return model_fn_lib.ModelFnOps(mode=mode,
                                   predictions=predictions,
                                   loss=loss,
                                   train_op=train_op,
                                   eval_metric_ops=eval_metric_ops)


def is_training(mode):
    return mode == tf.estimator.ModeKeys.TRAIN


def batch_norm(inputs, mode):
    return slim.batch_norm(inputs, is_training=is_training(mode))


def l2_normalize(inputs, axis):
    return tf.nn.l2_normalize(inputs, axis)


def is_predict(mode):
    return mode == ModeKeys.INFER


def is_evaluation(mode):
    return mode == ModeKeys.EVAL


def create_metric(values):
    return tf.metrics.mean(values)


def format_labels(labels, target_type, num_classes):
    if target_type == "sparse":
        return dense_to_sparse(labels, eos_token=num_classes)
    return labels


def feed(features, layer, mode):
    return _feed_to_layer(features, layer, mode)


def _feed_to_layer(inputs, layer, mode):
    layer_type = layer["layer_type"]
    if layer_type == "input_layer":
        return reshape(inputs, layer["shape"], layer["name"])
    if layer_type == "conv2d":
        return conv2d(inputs, num_filters=layer["num_filters"],
                      kernel=layer["kernel_size"])
    if layer_type == "max_pool2d":
        return max_pool2d(inputs, kernel=layer["pool_size"])
    if layer_type == "mdrnn":
        return mdrnn(inputs, num_hidden=layer["num_hidden"],
                     cell_type=layer["cell_type"])
    if layer_type == "dropout":
        return dropout(inputs, keep_prob=layer["keep_prob"],
                       mode=mode)
    if layer_type == "collapse_to_rnn_dims":
        return collapse_to_rnn_dims(inputs)
    if layer_type == "convert_to_ctc_dims":
        return convert_to_ctc_dims(inputs, num_classes=layer["num_classes"],
                                   num_steps=inputs.shape[1],
                                   num_hidden_units=inputs.shape[-1])
    if layer_type == "l2_normalize":
        return l2_normalize(inputs, layer["axis"])
    raise NotImplementedError(layer_type + " layer not implemented.")


def _get_sequence_lengths(inputs):
    dims = tf.stack([tf.shape(inputs)[1]])
    return tf.fill(dims, inputs.shape[0])


def get_output(inputs, output_layer):
    if output_layer == "ctc_decoder":
        decoded, _ = ctc_beam_search_decoder(inputs,
                                             _get_sequence_lengths(inputs))
        return sparse_to_dense(decoded, name="output")
    raise NotImplementedError(output_layer + " not implemented")


def get_loss(loss, labels, inputs):
    if loss == "ctc":
        return ctc_loss(labels=labels,
                        inputs=inputs,
                        sequence_length=_get_sequence_lengths(inputs))
    raise NotImplementedError(loss + " loss not implemented")


def get_metric(metrics, y_pred, y_true):
    metrics_dict = {}
    for metric in metrics:
        if metric == "label_error_rate":
            y_pred, _ = ctc_beam_search_decoder(y_pred,
                                                _get_sequence_lengths(y_pred))
            ler = label_error_rate(y_pred, y_true)
            add_to_summary(metric, ler)
            metrics_dict[metric] = create_metric(ler)
        else:
            raise NotImplementedError(metric + " metric not implemented")
    return metrics_dict


def model_fn(features, labels, mode, params):
    features = features["x"]

    network = params["network"]
    target_type = params["target_type"]
    metrics = params["metrics"]
    output_layer = params["output_layer"]
    loss = params["loss"]
    learning_rate = params["learning_rate"]
    optimizer = params["optimizer"]
    num_classes = params["num_classes"]

    labels = format_labels(labels, target_type, num_classes)

    for layer in network:
        features = feed(features, layer, mode)

    outputs = get_output(features, output_layer)
    predictions = {
        "outputs": outputs
    }
    if is_predict(mode):
        return create_model_fn(mode, predictions=predictions)

    loss = get_loss(loss, labels=labels, inputs=features)
    metrics = get_metric(metrics, y_pred=features, y_true=labels)

    if is_evaluation(mode):
        return create_model_fn(mode, predictions=predictions, loss=loss,
                               eval_metric_ops=metrics)

    assert is_training(mode)

    train_op = create_train_op(loss,
                               learning_rate=learning_rate,
                               optimizer=optimizer)
    return create_model_fn(mode,
                           predictions=predictions,
                           loss=loss,
                           train_op=train_op)

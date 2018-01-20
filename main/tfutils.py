import tensorflow as tf

from optimizer_enum import Optimizers
from tensorflow.contrib import rnn
from tensorflow.contrib import grid_rnn
from tensorflow.contrib.ndlstm.python import lstm2d
from tensorflow.contrib import slim
from tensorflow.contrib import learn

def ctc_loss(inputs, labels, sequence_length,
             preprocess_collapse_repeated_labels=True,
             ctc_merge_repeated=True,
             inputs_are_time_major=True):
    return tf.nn.ctc_loss(inputs=inputs, labels=labels, sequence_length=sequence_length,
                          preprocess_collapse_repeated=preprocess_collapse_repeated_labels,
                          ctc_merge_repeated=ctc_merge_repeated,
                          time_major=inputs_are_time_major)

def input_data(shape, name: str = 'InputData', input_type='float32'):
    return tf.placeholder(shape=shape, dtype=_get_type(input_type), name=name)

def reshape(tensor: tf.Tensor, new_shape: list):
    return tf.reshape(tensor, new_shape, name="reshape")

def stack_bidirectional_lstm(inputs, num_hidden_list):
    lstm_fw_cells = [rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) for num_hidden in num_hidden_list]
    lstm_bw_cells = [rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) for num_hidden in num_hidden_list]
    return tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cells, lstm_bw_cells, inputs,
                                                          dtype=tf.float32)[0]
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

def label_error_rate(y_pred, y_true):
    return tf.reduce_mean(tf.edit_distance(tf.cast(y_pred, tf.int32), y_true))

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

def sparse_input_data(input_type=tf.int32):
    return tf.sparse_placeholder(dtype=input_type)

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

def _get_type(type_str):
    if type_str == 'int32':
        return tf.int32
    return tf.float32

def get_shape(tensor):
    return tf.shape(tensor)

def initialize_variable(initial_value, name, is_trainable):
    return tf.Variable(initial_value, name=name, trainable=is_trainable)

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

def train(model, model_dir, input_fn, monitors=None):
    classifier = learn.Estimator(model_fn=model.model_fn, params=model.params, model_dir=model_dir)
    classifier.fit(input_fn=input_fn, monitors=monitors)

def freeze_graph(model_dir, output_node_names):
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node.")
        return -1

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

        saver.restore(sess, input_checkpoint)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names
        )

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
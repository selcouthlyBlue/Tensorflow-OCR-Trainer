import tensorflow as tf

from tensorboard import main as tb

from trainer.backend.tf import layers
from trainer.backend.GraphKeys import LayerTypes


def get_sequence_lengths(inputs):
    used = tf.sign(tf.reduce_max(tf.abs(inputs), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def feed(inputs, layer, is_training):
    layer_type = layer["layer_type"]
    if layer_type == LayerTypes.CONV2D.value:
        return layers.conv2d(inputs, num_filters=layer["num_filters"],
                             kernel_size=layer["kernel_size"],
                             activation=layer.get("activation"),
                             padding=layer.get("padding"),
                             scope=layer.get("name"))
    if layer_type == LayerTypes.MAX_POOL2D.value:
        return layers.max_pool2d(inputs, pool_size=layer["pool_size"],
                                 padding=layer.get("padding"),
                                 stride=layer.get("stride"),
                                 scope=layer.get("name"))
    if layer_type == LayerTypes.BIRNN.value:
        return layers.bidirectional_rnn(inputs, num_hidden=layer["num_hidden"],
                                        cell_type=layer.get("cell_type"),
                                        activation=layer.get("activation"),
                                        scope=layer.get("name"))
    if layer_type == LayerTypes.MDRNN.value:
        return layers.mdrnn(inputs, num_hidden=layer["num_hidden"],
                            cell_type=layer.get("cell_type"),
                            activation=layer.get("activation"),
                            scope=layer.get("name"))
    if layer_type == LayerTypes.DROPOUT.value:
        return layers.dropout(inputs, keep_prob=layer["keep_prob"],
                              is_training=is_training,
                              scope=layer.get("name"))
    if layer_type == LayerTypes.COLLAPSE_TO_RNN_DIMS.value:
        return layers.collapse_to_rnn_dims(inputs)
    if layer_type == LayerTypes.L2_NORMALIZE.value:
        return layers.l2_normalize(inputs, [1, 2])
    if layer_type == LayerTypes.BATCH_NORM.value:
        return layers.batch_norm(inputs, is_training=is_training)
    raise NotImplementedError(layer_type + " layer not implemented.")


def dense_to_sparse(tensor, token_to_ignore=0):
    indices = tf.where(tf.not_equal(tensor, tf.constant(token_to_ignore, dtype=tensor.dtype)))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def visualize(model, host):
    tf.flags.FLAGS.logdir = model
    tf.flags.FLAGS.host = host
    tb.main()

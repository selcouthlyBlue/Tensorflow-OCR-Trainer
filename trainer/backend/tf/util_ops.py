import tensorflow as tf

from trainer.backend.tf import layers


def get_sequence_lengths(inputs):
    dims = tf.stack([tf.shape(inputs)[1]])
    sequence_length = tf.fill(dims, inputs.shape[0])
    return sequence_length


def feed(features, layer, is_training):
    return _feed_to_layer(features, layer, is_training)


def _feed_to_layer(inputs, layer, is_training):
    layer_type = layer["layer_type"]
    if layer_type == "reshape":
        return layers.reshape(inputs, layer["shape"], layer.get("name"))
    if layer_type == "conv2d":
        return layers.conv2d(inputs, num_filters=layer["num_filters"],
                             kernel=layer["kernel_size"],
                             activation=layer.get("activation"),
                             padding=layer.get("padding"),
                             scope=layer.get("name"))
    if layer_type == "max_pool2d":
        return layers.max_pool2d(inputs, kernel=layer["pool_size"],
                                 padding=layer.get("padding"),
                                 stride=layer.get("stride"),
                                 scope=layer.get("name"))
    if layer_type == "birnn":
        return layers.bidirectional_rnn(inputs, num_hidden=layer["num_hidden"],
                                        cell_type=layer.get("cell_type"),
                                        activation=layer.get("activation"),
                                        scope=layer.get("name"))
    if layer_type == "mdrnn":
        return layers.mdrnn(inputs, num_hidden=layer["num_hidden"],
                            cell_type=layer.get("cell_type"),
                            activation=layer.get("activation"),
                            scope=layer.get("name"))
    if layer_type == "dropout":
        return layers.dropout(inputs, keep_prob=layer["keep_prob"],
                              is_training=is_training,
                              scope=layer.get("name"))
    if layer_type == "collapse_to_rnn_dims":
        return layers.collapse_to_rnn_dims(inputs)
    if layer_type == "l2_normalize":
        return layers.l2_normalize(inputs, layer["axis"])
    if layer_type == 'batch_norm':
        return layers.batch_norm(inputs, is_training=is_training)
    raise NotImplementedError(layer_type + " layer not implemented.")


def dense_to_sparse(tensor, token_to_ignore=0):
    indices = tf.where(tf.not_equal(tensor, tf.constant(token_to_ignore, dtype=tensor.dtype)))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)

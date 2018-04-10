import tensorflow as tf

from six.moves import xrange
from tensorboard import main as tb

from trainer.backend.tf import layers
from trainer.backend.GraphKeys import LayerTypes



def get_sequence_lengths(inputs):
    dims = tf.shape(inputs)[1]
    sequence_length = tf.fill([dims], inputs.shape[0])
    return sequence_length


def feed(features, layer, is_training):
    return _feed_to_layer(features, layer, is_training)


def _feed_to_layer(inputs, layer, is_training):
    layer_type = layer["layer_type"]
    if layer_type == LayerTypes.CONV2D.value:
        return layers.conv2d(inputs, num_filters=layer["num_filters"],
                             kernel=layer["kernel_size"],
                             activation=layer.get("activation"),
                             padding=layer.get("padding"),
                             scope=layer.get("name"))
    if layer_type == LayerTypes.MAX_POOL2D.value:
        return layers.max_pool2d(inputs, kernel=layer["pool_size"],
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
                            kernel_size=layer.get("kernel_size"),
                            scope=layer.get("name"))
    if layer_type == LayerTypes.DROPOUT.value:
        return layers.dropout(inputs, keep_prob=layer["keep_prob"],
                              is_training=is_training,
                              scope=layer.get("name"))
    if layer_type == LayerTypes.COLLAPSE_TO_RNN_DIMS.value:
        return layers.collapse_to_rnn_dims(inputs)
    if layer_type == LayerTypes.L2_NORMALIZE.value:
        return layers.l2_normalize(inputs, layer["axis"])
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


# see https://gist.github.com/moodoki/e37a85fb0258b045c005ca3db9cbc7f6
def freeze(checkpoint_dir, output_nodes=None,
           output_graph_filename='frozen-graph.pb'):
    if output_nodes is None:
        output_nodes = ["output"]
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    print(input_checkpoint)

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()

    input_graph_def = graph.as_graph_def()

    _fix_batch_norm_nodes(input_graph_def)

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = _freeze_variables(input_graph_def, output_nodes, sess)

        _write_graph(output_graph_def, output_graph_filename)
        print("%d ops in the final graph." % len(output_graph_def.node))


def _write_graph(output_graph_def, output_graph_filename):
    with tf.gfile.GFile(output_graph_filename, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def _freeze_variables(input_graph_def, output_nodes, sess):
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, input_graph_def,
        output_nodes
    )
    return output_graph_def


def _fix_batch_norm_nodes(input_graph_def):
    for node in input_graph_def.node:
        node_op = node.op
        if node_op == 'RefSwitch':
            node.op = 'Switch'

            input_node = node.input
            for index in xrange(len(input_node)):
                if 'moving' in input_node[index]:
                    input_node[index] = input_node + '/read'
        elif node_op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

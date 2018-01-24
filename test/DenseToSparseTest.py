import tensorflow as tf
from tensorflow.contrib import grid_rnn
from tfutils import dense_to_sparse


class DenseToSparseTest(tf.test.TestCase):
    def test_convert_dense_tensor_to_sparse_tensor(self):
        dense = tf.placeholder(tf.int32)
        sparse = dense_to_sparse(dense)
        sparse_converted_to_dense = tf.sparse_tensor_to_dense(sparse)
        self.assertTrue(dense.shape == sparse_converted_to_dense.shape)

    def test_feed_sparse_from_dense_to_ctc_loss(self):
        input_layer = tf.placeholder(tf.float32, [None, 1596, 48])
        dense = tf.placeholder(tf.int32, [None])
        sparse = dense_to_sparse(dense)

        cell_fw = grid_rnn.Grid2LSTMCell(num_units=128)
        cell_bw = grid_rnn.Grid2LSTMCell(num_units=128)
        bidirectional_grid_rnn = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_layer, dtype=tf.float32)
        outputs = tf.reshape(bidirectional_grid_rnn[0], [-1, 256])

        W = tf.Variable(tf.truncated_normal([256,
                                             80],
                                            stddev=0.1, dtype=tf.float32), name='W')
        b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[80], name='b'))

        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [tf.shape(input_layer)[0], -1, 80])
        logits = tf.transpose(logits, (1, 0, 2))

        loss = tf.nn.ctc_loss(inputs=logits, labels=sparse, sequence_length=[2])


if __name__ == '__main__':
    tf.test.main()

import tensorflow as tf
import numpy as np

from backend.tf.util_ops import dense_to_sparse


class DenseToSparseTest(tf.test.TestCase):
    def test_convert_dense_tensor_to_sparse_tensor(self):
        dense_array = np.array([1, 2, 3]).astype(np.int32)
        dense = tf.placeholder(tf.int32, shape=[None])
        sparse = dense_to_sparse(dense)
        sparse_converted_to_dense = tf.sparse_tensor_to_dense(sparse)
        with self.test_session() as sess:
            dense_from_sparse = sess.run(sparse_converted_to_dense, feed_dict={dense: dense_array})
        self.assertAllEqual(dense_from_sparse, dense_array)


if __name__ == '__main__':
    tf.test.main()

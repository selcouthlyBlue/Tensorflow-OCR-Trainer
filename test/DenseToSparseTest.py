import tensorflow as tf

from backend.tf.util_ops import dense_to_sparse


class DenseToSparseTest(tf.test.TestCase):
    def test_convert_dense_tensor_to_sparse_tensor(self):
        dense = tf.placeholder(tf.int32)
        sparse = dense_to_sparse(dense)
        sparse_converted_to_dense = tf.sparse_tensor_to_dense(sparse)
        self.assertTrue(dense.shape == sparse_converted_to_dense.shape)


if __name__ == '__main__':
    tf.test.main()

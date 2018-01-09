import dataset_utils
import tensorflow as tf
import tflearn

from tensorflow.contrib import grid_rnn


def main(_):
    image_paths, labels = dataset_utils.read_dataset_list('../test/dummy_labels_file.txt')
    data_dir = "../test/dummy_data/"
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension='png')
    print('Done reading images')
    images = dataset_utils.resize(images, (1596, 48))
    images = dataset_utils.transpose(images)
    labels = dataset_utils.encode(labels)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=0.5, labels=labels)
    y_train = dataset_utils.convert_to_sparse(y_train)
    y_test = dataset_utils.convert_to_sparse(y_test)

    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, None, 48])
        label_indices = tf.placeholder(tf.int64)
        label_shape = tf.placeholder(tf.int64)
        label_values = tf.placeholder(tf.int32)
        Y = tf.SparseTensor(indices=label_indices,
                            dense_shape=label_shape,
                            values=label_values)
        seq_lens = tf.placeholder(tf.int32, [None])

        def dnn(x):
            cell_fw = grid_rnn.Grid2LSTMCell(num_units=128)
            cell_bw = grid_rnn.Grid2LSTMCell(num_units=128)
            bidirectional_grid_rnn = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32)
            outputs = tf.reshape(bidirectional_grid_rnn[0], [-1, 256])

            W = tf.Variable(tf.truncated_normal([256,
                                                 80],
                                                stddev=0.1, dtype=tf.float32), name='W')
            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[80], name='b'))

            logits = tf.matmul(outputs, W) + b
            logits = tf.reshape(logits, [tf.shape(x)[0], -1, 80])
            logits = tf.transpose(logits, (1, 0, 2))
            return logits

        net = dnn(X)
        cost = tf.reduce_mean(tf.nn.ctc_loss(inputs=net, labels=Y, sequence_length=seq_lens))
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.5)

        train_op = tflearn.TrainOp(loss=cost, optimizer=optimizer)
        trainer = tflearn.Trainer(train_ops=train_op)

        trainer.fit(feed_dicts={X: x_train, label_indices: y_train[0], label_values: y_train[1], label_shape: y_train[2], seq_lens: dataset_utils.get_seq_lens(x_train)},
                    val_feed_dicts={X: x_test, label_indices: y_test[0], label_values: y_test[1], label_shape: y_test[2], seq_lens: dataset_utils.get_seq_lens(x_test)},
                    n_epoch=1)


if __name__ == '__main__':
    tf.app.run(main=main)

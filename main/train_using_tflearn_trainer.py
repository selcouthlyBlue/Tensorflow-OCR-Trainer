import dataset_utils
import tensorflow as tf
import tflearn
import tfutils as network_utils

from tensorflow.contrib import grid_rnn

from optimizer_enum import Optimizers


def main(_):
    image_paths, labels = dataset_utils.read_dataset_list('../test/dummy_labels_file.txt')
    data_dir = "../test/dummy_data/"
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension='png')
    print('Done reading images')
    images = dataset_utils.resize(images, (1596, 48))
    images = dataset_utils.transpose(images)
    labels = dataset_utils.encode(labels)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=0.5, labels=labels)

    num_hidden_units = 128
    num_classes = 80
    num_features = 48
    optimizer_name = Optimizers.MOMENTUM
    learning_rate = 0.001

    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, None, num_features])
        Y = tf.placeholder(tf.int32)

        sparse_Y = network_utils.dense_to_sparse(Y, num_classes)
        seq_lens = tf.placeholder(tf.int32, [None])

        def dnn(x):
            layer = network_utils.bidirectional_grid_lstm(inputs=x, num_hidden=num_hidden_units)
            layer = network_utils.get_time_major(inputs=layer, batch_size=network_utils.get_shape(x)[0],
                                                 num_classes=num_classes, num_hidden_units=num_hidden_units * 2)
            return layer

        net = dnn(X)
        cost = network_utils.cost(network_utils.ctc_loss(inputs=net, labels=sparse_Y, sequence_length=seq_lens))
        optimizer = network_utils.get_optimizer(learning_rate=learning_rate, optimizer_name=optimizer_name)

        train_op = tflearn.TrainOp(loss=cost, optimizer=optimizer)
        trainer = tflearn.Trainer(train_ops=train_op)

        trainer.fit(feed_dicts={X: x_train, Y: y_train, seq_lens: dataset_utils.get_seq_lens(x_train)},
                    val_feed_dicts={X: x_test, Y: y_test, seq_lens: dataset_utils.get_seq_lens(x_test)},
                    n_epoch=1)


if __name__ == '__main__':
    tf.app.run(main=main)
import time

import os

import dataset_utils
import tensorflow as tf

from model import Model
from optimizer_enum import Optimizers


def main(_):
    image_paths, labels = dataset_utils.read_dataset_list('../test/dummy_labels_file.txt')
    data_dir = "../test/dummy_data/"
    log_dir = "log/train/"
    num_epochs = 1
    batch_size = 1
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension='png')
    print('Done reading images')
    images = dataset_utils.resize(images, (1596, 48))
    images = dataset_utils.transpose(images)
    labels = dataset_utils.encode(labels)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=0.5, labels=labels)

    with tf.Graph().as_default():
        bi_lstm_model = Model()
        logits = bi_lstm_model.inference()
        loss = bi_lstm_model.loss(logits)
        cost = bi_lstm_model.cost(logits)
        train_op = bi_lstm_model.training(loss=loss, optimizer_name=Optimizers.MOMENTUM)
        label_error_rate = bi_lstm_model.label_error_rate(logits)
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for current_epoch in range(num_epochs):
            train_cost = train_label_error_rate = 0
            start = time.time()

            for batch in range(len(x_train)//batch_size):
                feed = {bi_lstm_model.inputs: x_train,
                        bi_lstm_model.seq_lens: dataset_utils.get_seq_lens(x_train),
                        bi_lstm_model.labels: dataset_utils.convert_to_sparse(y_train)}

                batch_cost, _ = sess.run([loss, train_op], feed)
                train_cost += batch_cost * batch_size
                train_label_error_rate += sess.run(label_error_rate, feed) * batch_size

            train_cost /= len(x_train)
            train_label_error_rate /= len(x_train)

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
            print(log.format(current_epoch + 1, num_epochs, train_cost[0], train_label_error_rate,
                             time.time() - start))

            if current_epoch % 10 == 0:
                print("Validating...")
                val_feed = {bi_lstm_model.inputs: x_test,
                            bi_lstm_model.seq_lens: dataset_utils.get_seq_lens(x_test),
                            bi_lstm_model.labels: dataset_utils.convert_to_sparse(y_test)}
                val_cost, val_label_error_rate = sess.run([cost, label_error_rate], val_feed)
                log = "val_cost = {:.3f}, val_ler = {:.3f}".format(val_cost, val_label_error_rate)
                print(log)
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=current_epoch)


if __name__ == '__main__':
    tf.app.run(main=main)

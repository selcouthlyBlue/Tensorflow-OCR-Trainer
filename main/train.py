import dataset_utils
import tensorflow as tf
import tflearn

from model import Model

if __name__ == '__main__':
    image_paths, labels = dataset_utils.read_dataset_list('../test/dummy_labels_file.txt')
    data_dir = "../test/dummy_data/"
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension='png')
    images = dataset_utils.resize(images, (1596, 48))
    labels = dataset_utils.encode(labels)
    x_train, y_train, x_test, y_test = dataset_utils.split(features=images, test_size=0.5, labels=labels)
    x_train_seq_lens = dataset_utils.get_seq_lens(x_train)
    x_test_seq_lens = dataset_utils.get_seq_lens(x_test)
    y_train = dataset_utils.convert_to_sparse(y_train)
    y_test = dataset_utils.convert_to_sparse(y_test)
    bi_lstm_model = Model()
    with tf.Graph().as_default():
        loss, label_error_rate = bi_lstm_model.loss()
        optimizer = bi_lstm_model.optimize('adam', 0.0001)
        train_op = tflearn.TrainOp(loss=loss, optimizer=optimizer, metric=label_error_rate, batch_size=128)
        trainer = tflearn.Trainer(train_ops=train_op, tensorboard_verbose=1)
        trainer.fit({bi_lstm_model.inputs: x_train, bi_lstm_model.seq_lens: x_train_seq_lens, bi_lstm_model.labels: y_train},
                    val_feed_dicts={bi_lstm_model.inputs: x_train, bi_lstm_model.seq_lens: x_test_seq_lens,  bi_lstm_model.labels: y_train})

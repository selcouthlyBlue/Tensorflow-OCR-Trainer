import dataset_utils
import tensorflow as tf

from tflearn import BasicLSTMCell, bidirectional_rnn, regression, input_data, DNN


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
    net = input_data([None, 1596, 48])
    net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
    net = regression(net, optimizer='adam', loss='ctc_loss')

    model = DNN(net, tensorboard_verbose=0)
    model.fit(x_train, y_train, show_metric=True, batch_size=1)


if __name__ == '__main__':
    tf.app.run(main=main)

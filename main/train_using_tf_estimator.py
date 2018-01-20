import dataset_utils
import tensorflow as tf
import numpy as np
import tfutils

from tensorflow.contrib import learn

from GridRNNCTCModel import GridRNNModelFn
from CNNMDLSTMCTCModel import CNNMDLSTMCTCModelFn
from optimizer_enum import Optimizers

tf.logging.set_verbosity(tf.logging.INFO)

def prepare_dataset(labels_file, data_dir, image_extension, desired_image_size, test_set_fraction):
    image_paths, labels = dataset_utils.read_dataset_list(labels_file)
    data_dir = data_dir
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension=image_extension)
    print('Done reading images')
    images = dataset_utils.resize(images, desired_image_size)
    images = dataset_utils.transpose(images)
    labels = dataset_utils.encode(labels)
    labels = dataset_utils.pad(labels)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=test_set_fraction, labels=labels)
    return x_test, x_train, y_test, y_train


def main(_):
    desired_image_width = 1596
    desired_image_height = 48
    starting_filter_size = 16
    labels_file = '../test/dummy_labels_file.txt'
    data_dir = "../test/dummy_data/"
    test_set_size = 0.5
    learning_rate = 0.001
    number_of_epochs_to_pass_before_validation = 5
    optimizer = Optimizers.MOMENTUM
    checkpoint_dir = 'grid_rnn_ocr_model'
    architecture = 'CNNMDLSTM'
    batch_size = 1
    num_channels = 1
    num_hidden_units=128

    x_test, x_train, y_test, y_train = prepare_dataset(
        labels_file=labels_file,
        data_dir=data_dir,
        image_extension='png',
        desired_image_size=(desired_image_width, desired_image_height),
        test_set_fraction=test_set_size
    )

    train_input_fn = create_input_fn(x_train, y_train)
    validation_input_fn = create_input_fn(x_test, y_test)

    validation_monitor = learn.monitors.ValidationMonitor(
        input_fn=validation_input_fn,
        every_n_steps=number_of_epochs_to_pass_before_validation
    )

    if architecture == 'CNNMDLSTM':
        model = CNNMDLSTMCTCModelFn(
            input_shape=[batch_size, desired_image_width, desired_image_height, num_channels],
            starting_filter_size=starting_filter_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            num_classes=81
        )
    else:
        model = GridRNNModelFn(
            input_shape=[batch_size, desired_image_width, desired_image_height],
            num_hidden_units=num_hidden_units,
            learning_rate=learning_rate,
            optimizer=optimizer,
            num_classes=81
        )

    tfutils.train(model, "checkpoint/" + checkpoint_dir, train_input_fn, monitors=[validation_monitor])
    tfutils.freeze_graph("checkpoint/" + checkpoint_dir, ["output"])


def create_input_fn(x, y, batch_size=1, num_epochs=1, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(x),
           "seq_lens": np.array(dataset_utils.get_seq_lens(x))},
        y=np.array(y),
        num_epochs=num_epochs,
        shuffle=shuffle,
        batch_size=batch_size
    )


if __name__ == '__main__':
    tf.app.run(main=main)
import dataset_utils
import tensorflow as tf
import numpy as np
import tfutils

from tensorflow.contrib import learn

from GridRNNModelFn import GridRNNModelFn
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
    labels = dataset_utils.pad(labels, blank_token_index=80)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=test_set_fraction, labels=labels)
    return x_test, x_train, y_test, y_train


def main(_):
    desired_image_width = 1596
    desired_image_height = 48

    x_test, x_train, y_test, y_train = prepare_dataset(
        labels_file='../test/dummy_labels_file.txt',
        data_dir="../test/dummy_data/",
        image_extension='png',
        desired_image_size=(desired_image_width, desired_image_height),
        test_set_fraction=0.5
    )

    train_input_fn = create_input_fn(x_train, y_train)
    validation_input_fn = create_input_fn(x_test, y_test)

    validation_monitor = learn.monitors.ValidationMonitor(
        input_fn=validation_input_fn,
        every_n_steps=5
    )

    model = GridRNNModelFn(num_time_steps=desired_image_width, num_features=desired_image_height, num_hidden_units=128, num_classes=80,
                           learning_rate=0.001, optimizer=Optimizers.MOMENTUM)

    tfutils.train(model, "/tmp/grid_rnn_ocr_model", train_input_fn, monitors=[validation_monitor])


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

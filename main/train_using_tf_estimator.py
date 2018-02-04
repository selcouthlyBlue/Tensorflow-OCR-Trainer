import dataset_utils
import tensorflow as tf
import numpy as np

from architecture_enum import Architectures
from GridRNNCTCModel import GridRNNCTCModel
from CNNMDLSTMCTCModel import CNNMDLSTMCTCModel
from tfutils import run_experiment, input_fn

tf.logging.set_verbosity(tf.logging.INFO)


def train(labels_file, data_dir, desired_image_height, desired_image_width, architecture, num_hidden_units, optimizer, learning_rate,
          test_fraction, validation_steps=5, num_epochs=1, batch_size=1, labels_delimiter=' '):
    image_paths, labels = dataset_utils.read_dataset_list(labels_file, delimiter=labels_delimiter)
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension='png')
    images = dataset_utils.binarize(images)
    images = dataset_utils.resize(images,
                                  desired_height=desired_image_height,
                                  desired_width=desired_image_width)
    images = dataset_utils.images_as_float32(images)

    checkpoint_dir = "checkpoint/"

    checkpoint_dir, images, model = initialize_model(architecture, checkpoint_dir, desired_image_height,
                                                     desired_image_width, images, learning_rate, num_hidden_units, optimizer)

    labels = dataset_utils.encode(labels)
    labels = dataset_utils.pad(labels, blank_token_index=80)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=test_fraction, labels=labels)

    print("Number of training samples:", len(x_train))
    print("Number of validation samples:", len(x_test))

    train_input_fn = input_fn(
        x_feed_dict={"x": np.array(x_train),
                     "seq_lens": dataset_utils.set_seq_lens(desired_image_width, len(x_train))},
        y=np.array(y_train, dtype=np.int32),
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    validation_input_fn = input_fn(
        x_feed_dict={"x": np.array(x_test),
                     "seq_lens": dataset_utils.set_seq_lens(desired_image_width, len(x_test))},
        y=np.array(y_test, dtype=np.int32),
        batch_size=batch_size,
        shuffle=False
    )

    run_experiment(model=model,
                   train_input_fn=train_input_fn,
                   checkpoint_dir=checkpoint_dir,
                   validation_input_fn=validation_input_fn,
                   validation_steps=validation_steps * (len(x_train)//batch_size))


def initialize_model(architecture, checkpoint_dir, desired_image_height,
                     desired_image_width, images, learning_rate, num_hidden_units, optimizer):
    if architecture == Architectures.CNNMDLSTM:
        model = CNNMDLSTMCTCModel(input_shape=[-1, desired_image_height, desired_image_width, 1],
                                  starting_filter_size=num_hidden_units,
                                  learning_rate=learning_rate, optimizer=optimizer, num_classes=80)
        checkpoint_dir += Architectures.CNNMDLSTM.value
    else:
        images = dataset_utils.transpose(images)
        model = GridRNNCTCModel(input_shape=[-1, desired_image_width, desired_image_height],
                                num_hidden_units=num_hidden_units, num_classes=80,
                                learning_rate=learning_rate, optimizer=optimizer)
        checkpoint_dir += Architectures.GRIDLSTM.value
    return checkpoint_dir, images, model


if __name__ == '__main__':
    tf.app.run(main=train)

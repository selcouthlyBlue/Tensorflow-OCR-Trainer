import dataset_utils
import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn

from architecture_enum import Architectures
from GridRNNCTCModel import GridRNNCTCModel
from CNNMDLSTMCTCModel import CNNMDLSTMCTCModel

tf.logging.set_verbosity(tf.logging.INFO)

def train(labels_file, data_dir, desired_image_size, architecture, num_hidden_units, optimizer, learning_rate,
          test_fraction, validation_steps=5, num_epochs=1, batch_size=1):
    image_paths, labels = dataset_utils.read_dataset_list(labels_file)
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension='png')
    images = dataset_utils.resize(images, desired_image_size)
    print('Done reading images')

    checkpoint_dir = "checkpoint/"

    checkpoint_dir, images, model = initialize_model(architecture, batch_size, checkpoint_dir, desired_image_size,
                                                     images, learning_rate, num_hidden_units, optimizer)

    labels = dataset_utils.encode(labels)
    labels = dataset_utils.pad(labels, blank_token_index=80)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=test_fraction, labels=labels)

    train_input_fn = create_input_fn(x_train, y_train, num_epochs=num_epochs, batch_size=batch_size)
    validation_input_fn = create_input_fn(x_test, y_test)

    classifier = learn.Estimator(model_fn=model.model_fn, params=model.params, model_dir=checkpoint_dir)
    classifier.fit(input_fn=train_input_fn)
    classifier.evaluate(input_fn=validation_input_fn)


def initialize_model(architecture, batch_size, checkpoint_dir, desired_image_size, images, learning_rate,
                     num_hidden_units, optimizer):
    if architecture == Architectures.CNNMDLSTM:
        model = CNNMDLSTMCTCModel(input_shape=[batch_size, desired_image_size[0], desired_image_size[1], 1],
                                  starting_filter_size=num_hidden_units,
                                  learning_rate=learning_rate, optimizer=optimizer, num_classes=80)
        checkpoint_dir += Architectures.CNNMDLSTM.value
    else:
        images = dataset_utils.transpose(images)
        model = GridRNNCTCModel(input_shape=[batch_size, desired_image_size[0], desired_image_size[1]],
                                num_hidden_units=num_hidden_units, num_classes=80,
                                learning_rate=learning_rate, optimizer=optimizer)
        checkpoint_dir += Architectures.GRIDLSTM.value
    return checkpoint_dir, images, model


def create_input_fn(x, y, num_epochs=1, shuffle=True, batch_size=1):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(x),
           "seq_lens": dataset_utils.get_seq_lens(x)},
        y=np.array(y),
        num_epochs=num_epochs,
        shuffle=shuffle,
        batch_size=batch_size
    )
    return train_input_fn


if __name__ == '__main__':
    tf.app.run(main=train)

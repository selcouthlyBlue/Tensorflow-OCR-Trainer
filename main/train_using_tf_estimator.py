import dataset_utils
import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn

from GridRNNModelFn import GridRNNModelFn
from CNNMDLSTMCTCModelFn import CNNMDLSTMCTCModelFn
from optimizer_enum import Optimizers

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    image_paths, labels = dataset_utils.read_dataset_list('../test/dummy_labels_file.txt')
    data_dir = "../test/dummy_data/"
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension='png')
    images = dataset_utils.resize(images, (1596, 48))
    print('Done reading images')

    checkpoint_dir = "checkpoint/"
    architecture = "cnnmdlstm"

    if architecture == "cnnmdlstm":
        model = CNNMDLSTMCTCModelFn(input_shape=[1, 1596, 48, 1], starting_filter_size=16,
                                    learning_rate=0.001, optimizer=Optimizers.MOMENTUM, num_classes=80)
        checkpoint_dir += "cnnmdlstm"
    else:
        images = dataset_utils.transpose(images)
        model = GridRNNModelFn(input_shape=[-1, 1596, 48], num_hidden_units=128, num_classes=80,
                               learning_rate=0.001, optimizer=Optimizers.MOMENTUM)
        checkpoint_dir += "gridlstm"

    labels = dataset_utils.encode(labels)
    labels = dataset_utils.pad(labels, blank_token_index=80)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=0.5, labels=labels)

    train_input_fn = create_input_fn(x_train, y_train)
    validation_input_fn = create_input_fn(x_test, y_test)

    validation_monitor = learn.monitors.ValidationMonitor(
        input_fn=validation_input_fn,
        every_n_steps=5
    )

    classifier = learn.Estimator(model_fn=model.model_fn, params=model.params, model_dir=checkpoint_dir)
    classifier.fit(input_fn=train_input_fn, monitors=[validation_monitor])


def create_input_fn(x, y):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(x),
           "seq_lens": dataset_utils.get_seq_lens(x)},
        y=np.array(y),
        num_epochs=1,
        shuffle=True,
        batch_size=1
    )
    return train_input_fn


if __name__ == '__main__':
    tf.app.run(main=main)

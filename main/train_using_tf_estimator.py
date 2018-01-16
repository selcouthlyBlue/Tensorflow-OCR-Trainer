import dataset_utils
import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn

from GridRNNModelFn import GridRNNModelFn
from optimizer_enum import Optimizers

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    image_paths, labels = dataset_utils.read_dataset_list('../test/dummy_labels_file.txt')
    data_dir = "../test/dummy_data/"
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension='png')
    print('Done reading images')
    images = dataset_utils.resize(images, (1596, 48))
    images = dataset_utils.transpose(images)
    labels = dataset_utils.encode(labels)
    labels = dataset_utils.pad(labels, blank_token_index=80)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=0.5, labels=labels)
    x_train_seq_lens = dataset_utils.get_seq_lens(x_train)
    x_test_seq_lens = dataset_utils.get_seq_lens(x_test)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(x_train),
           "seq_lens": np.array(x_train_seq_lens)},
        y=np.array(y_train),
        num_epochs=1,
        shuffle=True,
        batch_size=1
    )

    validation_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(x_test),
           "seq_lens": np.array(x_test_seq_lens)},
        y=np.array(y_test),
        shuffle=True
    )

    validation_monitor = learn.monitors.ValidationMonitor(
        input_fn=validation_input_fn,
        every_n_steps=5
    )

    model = GridRNNModelFn(num_time_steps=1596, num_features=48, num_hidden_units=128, num_classes=80,
                           learning_rate=0.001, optimizer=Optimizers.MOMENTUM)

    classifier = learn.Estimator(model_fn=model.model_fn, params=model.params, model_dir="/tmp/grid_rnn_ocr_model")
    classifier.fit(input_fn=train_input_fn, monitors=[validation_monitor])


if __name__ == '__main__':
    tf.app.run(main=main)

import os

import numpy as np

import dataset_utils
from backend.tf.experiment_ops import run_experiment, input_fn


def train(model_config_file, labels_file, data_dir, desired_image_height,
          desired_image_width, labels_delimiter=' ', max_label_length=120,
          test_fraction=None, num_epochs=1, batch_size=1, validation_steps=1):
    image_paths, labels = dataset_utils.read_dataset_list(
        labels_file, delimiter=labels_delimiter)
    images = dataset_utils.read_images(data_dir=data_dir,
                                       image_paths=image_paths,
                                       image_extension='png')
    images = dataset_utils.resize(images,
                                  desired_height=desired_image_height,
                                  desired_width=desired_image_width)
    images = dataset_utils.binarize(images)
    images = dataset_utils.invert(images)
    images = dataset_utils.images_as_float32(images)

    labels = dataset_utils.encode(labels)
    labels = dataset_utils.pad(labels, blank_token_index=80,
                               max_label_length=max_label_length)

    x_train, x_test, y_train, y_test = dataset_utils.split(
        features=images, test_size=test_fraction, labels=labels)

    train_input_fn = input_fn(
        x_feed_dict={"x": np.array(x_train)},
        y=np.array(y_train, dtype=np.int32),
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    validation_input_fn = None
    if test_fraction:
        validation_input_fn = input_fn(
            x_feed_dict={"x": np.array(x_test)},
            y=np.array(y_test, dtype=np.int32),
            batch_size=batch_size,
            shuffle=False
        )

    filename, _ = os.path.splitext(model_config_file)
    model_name = filename.split('/')[-1]

    run_experiment(model_config_file=model_config_file,
                   train_input_fn=train_input_fn,
                   validation_input_fn=validation_input_fn,
                   checkpoint_dir="checkpoint/" + model_name,
                   validation_steps=validation_steps * (len(x_train)//batch_size))

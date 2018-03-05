import os
import json

import dataset_utils
from backend.tf.experiment_ops import run_experiment


def train(model_config_file, labels_file, data_dir, desired_image_height,
          desired_image_width, charset_file='../charsets/chars.txt',
          labels_delimiter=' ', max_label_length=120,
          test_fraction=None, num_epochs=1, batch_size=1,
          save_checkpoint_epochs=1):
    params = json.load(open(model_config_file, 'r'))

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
    classes = dataset_utils.get_characters_from(charset_file)
    images = dataset_utils.images_as_float32(images)
    labels = dataset_utils.encode(labels, classes)
    num_classes = len(classes) + 1
    labels = dataset_utils.pad(labels, max_label_length=max_label_length)

    filename, _ = os.path.splitext(model_config_file)
    model_name = filename.split('/')[-1]

    run_experiment(params=params,
                   features=images,
                   labels=labels,
                   num_classes=num_classes,
                   checkpoint_dir="checkpoint/" + str(model_name),
                   batch_size=batch_size,
                   num_epochs=num_epochs,
                   save_checkpoint_every_n_epochs=save_checkpoint_epochs,
                   test_fraction=test_fraction)

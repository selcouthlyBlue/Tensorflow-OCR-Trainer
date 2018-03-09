import os

from train_ocr import train
from werkzeug.utils import secure_filename

def start_training(train_params):
    train(
        model_config_file=train_params['model_config'],
        labels_file=train_params['labels_file'],
        data_dir=train_params['data_dir'],
        desired_image_height=train_params['desired_image_height'],
        desired_image_width=train_params['desired_image_width'],
        max_label_length=train_params['max_label_length'],
        num_epochs=train_params['num_epochs'],
        test_fraction=train_params['test_fraction'],
        batch_size=train_params['batch_size'],
        save_checkpoint_epochs=train_params['save_checkpoint_epochs']
    )

def get_directory_list(path):
    directory_names = os.listdir(path)
    return directory_names

def upload_dataset(images, labels_file, dataset_directory, dataset_name):
    dataset_path = _create_path(dataset_directory, dataset_name)
    os.makedirs(dataset_path)
    labels_file.save(_create_path(dataset_path,
                                  secure_filename(labels_file.filename)))
    for image in images:
        image.save(_create_path(dataset_path,
                                secure_filename(image.filename)))


def _create_path(*args):
    return os.path.join(*args)

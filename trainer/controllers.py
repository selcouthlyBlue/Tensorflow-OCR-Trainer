import os
import json
import csv
import shutil
import multiprocessing
import time

from werkzeug.utils import secure_filename
from flask import request
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from trainer import app
from trainer.backend import GraphKeys
from trainer.backend.dataset_utils import read_dataset_list
from trainer.backend.train_ocr import train_model
from trainer.backend.train_ocr import evaluate_model
from trainer.backend import visualize


def _allowed_labels_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() \
           in app.config['ALLOWED_LABELS_FILE_EXTENSIONS']


def _allowed_image_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() \
           in app.config['ALLOWED_IMAGE_EXTENSIONS']


def upload_dataset(images, labels_file):
    if (labels_file and _allowed_labels_file(labels_file.filename)) \
            and (images and _allowed_image_files(image.filename)
                 for image in images):
        _upload_dataset_files(images, labels_file)
        split_dataset(labels_file)
        return get('dataset_name') + " has been uploaded."
    return "An error occurred in uploading the dataset."


def _upload_dataset_files(images, labels_file):
    dataset_name = get('dataset_name')
    dataset_path = create_path(app.config['DATASET_DIRECTORY'], dataset_name)
    os.makedirs(dataset_path)
    labels_file.save(create_path(dataset_path,
                                 secure_filename(labels_file.filename)))
    for image in images:
        image.save(create_path(dataset_path,
                               secure_filename(image.filename)))


def get_dataset(dataset_name):
    return create_path(app.config['DATASET_DIRECTORY'], dataset_name)


def get_directory_list_from_config(directory_key):
    directory_names = os.listdir(app.config[directory_key])
    return directory_names


def _create_labels_file(filename, features, labels):
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n', delimiter=' ')
        writer.writerows(zip(features, labels))


def split_dataset(labels_file):
    dataset_path = create_path(app.config['DATASET_DIRECTORY'], get('dataset_name'))
    features, labels = read_dataset_list(create_path(dataset_path,
                                                     secure_filename(labels_file.filename)))
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=float(get('test_size')))
    _create_labels_file(create_path(dataset_path, 'train.csv'), x_train, y_train)
    _create_labels_file(create_path(dataset_path, 'test.csv'), x_test, y_test)


def create_path(*args):
    return os.path.join(*args)


def get_enum_values(enum_type):
    return [key.value.replace('"', '') for key in enum_type]


def _get_layer(layer_index):
    layer = OrderedDict()
    layer["layer_type"] = get(_create_network_key(layer_index, "layer_type"))
    layer_type = layer["layer_type"]
    if layer_type == GraphKeys.LayerTypes.CONV2D.value:
        layer["num_filters"] = get(_create_network_key(layer_index, "num_filters"))
        layer["kernel_size"] = get(_create_network_key(layer_index, "kernel_size"))
        layer["stride"] = get(_create_network_key(layer_index, "stride"))
        layer["padding"] = get(_create_network_key(layer_index, "padding"))
    elif layer_type == GraphKeys.LayerTypes.MAX_POOL2D.value:
        layer["pool_size"] = get(_create_network_key(layer_index, "pool_size"))
        layer["stride"] = get(_create_network_key(layer_index, "stride"))
        layer["padding"] = get(_create_network_key(layer_index, "padding"))
    elif layer_type == GraphKeys.LayerTypes.BIRNN.value:
        layer["num_hidden"] = get(_create_network_key(layer_index, "num_hidden"))
        layer["cell_type"] = get(_create_network_key(layer_index, "cell_type"))
        layer["activation"] = get(_create_network_key(layer_index, "activation"))
    elif layer_type == GraphKeys.LayerTypes.MDRNN.value:
        layer["num_hidden"] = get(_create_network_key(layer_index, "num_hidden"))
        layer["kernel_size"] = get(_create_network_key(layer_index, "kernel_size"))
        layer["cell_type"] = get(_create_network_key(layer_index, "cell_type"))
        layer["activation"] = get(_create_network_key(layer_index, "activation"))
    elif layer_type == GraphKeys.LayerTypes.DROPOUT.value:
        layer["keep_prob"] = get(_create_network_key(layer_index, "keep_prob"))
    elif layer_type == GraphKeys.LayerTypes.L2_NORMALIZE:
        layer["axis"] = get(_create_network_key(layer_index, "axis"))
    return layer


def get_architecture_file_contents(architecture_name):
    architecture_path = get_architecture_path(architecture_name)
    return json.load(open(architecture_path), object_pairs_hook=OrderedDict)


def delete_file(file_path):
    os.remove(file_path)


def delete_folder(folder_name):
    shutil.rmtree(folder_name)


def _create_network_key(layer_index, param):
    network_key = "network[" + str(layer_index) + "][" + param + "]"
    return network_key


def _generate_architecture_dict():
    resulting_dict = OrderedDict()
    network = []
    number_of_layers = len([value for key, value in request.form.items() if 'layer_type' in key.lower()])
    for layer_index in range(number_of_layers):
        layer = _get_layer(layer_index)
        network.append(layer)
    resulting_dict['network'] = network
    resulting_dict['output_layer'] = get("output_layer")
    return resulting_dict


def get_model_path(model):
    return create_path(app.config['MODELS_DIRECTORY'], model)


def visualize_model(model_name, host):
    model_path = get_model_path(model_name)
    visualization_task = multiprocessing.Process(target=visualize, args=(model_path, host))
    visualization_task.name = "visualize {}".format(model_path)
    visualization_task.start()


def get_visualization_link():
    time.sleep(20)
    return "http://localhost:6006"


def save_model_as_json():
    architecture_dict = _generate_architecture_dict()
    architecture_name = get('architecture_name')
    architecture_path = get_architecture_path(architecture_name)
    with open(architecture_path, 'w') as fp:
        json.dump(architecture_dict, fp, indent=4)
    return architecture_name + " has been created."


def get_architecture_path(architecture_name):
    return create_path(app.config['ARCHITECTURES_DIRECTORY'], architecture_name) + ".json"


def get(param):
    return request.form[param]


def getlist(param):
    return request.form.getlist(param)


def stop_running(task):
    for running_task in multiprocessing.active_children():
        if running_task.name == task:
            running_task.terminate()
            running_task.join()
            return task.capitalize() + " is terminated."
    return task.capitalize() + " was already terminated."


def get_running_tasks():
    return [running_task.name for running_task in multiprocessing.active_children()]


def test_task(dataset_name, charset_file,
              desired_image_size, model_name, batch_size=1,
              labels_delimiter=' '):
    dataset_dir = get_dataset(dataset_name)
    checkpoint_dir = get_model_path(model_name)
    architecture_params = json.load(open(create_path(checkpoint_dir, "architecture.json")), object_pairs_hook=OrderedDict)
    testing_task = multiprocessing.Process(target=evaluate_model,
                                           args=(architecture_params,
                                                 dataset_dir,
                                                 charset_file,
                                                 desired_image_size,
                                                 checkpoint_dir,
                                                 batch_size,
                                                 labels_delimiter))
    testing_task.start()
    return testing_task


def run_learning_task(task):
    running_task = None
    dataset_name = get('dataset_name')
    if task == 'training':
        running_task = train_task(get('architecture_name'),
                                  dataset_name,
                                  int(get('desired_image_size')),
                                  int(get('num_epochs')),
                                  int(get('checkpoint_epochs')),
                                  int(get('batch_size')),
                                  'charsets/chars.txt',
                                  float(get('learning_rate')),
                                  get('optimizer'),
                                  getlist('metrics'),
                                  get('loss'))
    elif task == 'testing':
        print(request.form)
        print(get('desired_image_size'))
        print(int(get('desired_image_size')))
        print(get('model_name'))
        running_task = test_task(dataset_name,
                                 'charsets/chars.txt',
                                 int(get('desired_image_size')),
                                 get('model_name'),
                                 1,
                                 ' ')
    time.sleep(30)
    running_task.name = task


def train_task(architecture_name,
               dataset_name,
               desired_image_size,
               num_epochs,
               checkpoint_epochs,
               batch_size,
               charset_file,
               learning_rate,
               optimizer,
               metrics,
               loss):
    dataset_dir = get_dataset(dataset_name)
    checkpoint_dir = get_model_path("model-" + time.strftime("%Y%m%d-%H%M%S"))
    os.mkdir(checkpoint_dir)
    model_architecture_path = _copy_architecture_to_model(architecture_name, checkpoint_dir)
    architecture_params = json.load(open(model_architecture_path), object_pairs_hook=OrderedDict)
    task = multiprocessing.Process(target=train_model,
                                   args=(
                                       architecture_params,
                                       dataset_dir,
                                       checkpoint_dir,
                                       learning_rate,
                                       metrics,
                                       loss,
                                       optimizer,
                                       desired_image_size,
                                       charset_file,
                                       ' ',
                                       num_epochs,
                                       batch_size,
                                       checkpoint_epochs
                                   ))
    task.start()
    return task


def _copy_architecture_to_model(architecture_name, checkpoint_dir):
    architecture_config_file = get_architecture_path(architecture_name)
    shutil.copy(architecture_config_file, checkpoint_dir)
    model_architecture_path = create_path(checkpoint_dir, "architecture.json")
    os.rename(create_path(checkpoint_dir, architecture_name) + ".json", model_architecture_path)
    return model_architecture_path

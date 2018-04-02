import os
import json
import csv
import subprocess
import time

from werkzeug.utils import secure_filename
from flask import request, flash
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from trainer import app
from trainer.backend import GraphKeys
from trainer.backend.dataset_utils import read_dataset_list


def get_directory_list(path):
    directory_names = os.listdir(path)
    return directory_names

def upload_dataset(images, labels_file, dataset_directory):
    dataset_path = create_path(dataset_directory, _get('dataset_name'))
    os.makedirs(dataset_path)
    labels_file.save(create_path(dataset_path,
                                 secure_filename(labels_file.filename)))
    for image in images:
        image.save(create_path(dataset_path,
                               secure_filename(image.filename)))
    flash(_get('dataset_name'), " uploaded.")


def _create_labels_file(filename, features, labels):
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n', delimiter=' ')
        writer.writerows(zip(features, labels))


def split_dataset(labels_file, dataset_directory):
    dataset_path = create_path(dataset_directory, _get('dataset_name'))
    features, labels = read_dataset_list(create_path(dataset_path,
                                         secure_filename(labels_file.filename)))
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=float(_get('test_size')))
    _create_labels_file(create_path(dataset_path, 'train.csv'), x_train, y_train)
    _create_labels_file(create_path(dataset_path, 'test.csv'), x_test, y_test)


def create_path(*args):
    return os.path.join(*args)


def get_enum_values(enum_type):
    return [key.value.replace('"', '') for key in enum_type]


def _get_layer(layer_index):
    layer = OrderedDict()
    layer["layer_type"] = _get(_create_network_key(layer_index, "layer_type"))
    layer_type = layer["layer_type"]
    if layer_type == GraphKeys.LayerTypes.CONV2D.value:
        layer["num_filters"] = _get(_create_network_key(layer_index, "num_filters"))
        layer["kernel_size"] = _get(_create_network_key(layer_index, "kernel_size"))
        layer["stride"] = _get(_create_network_key(layer_index, "stride"))
        layer["padding"] = _get(_create_network_key(layer_index, "padding"))
    elif layer_type == GraphKeys.LayerTypes.MAX_POOL2D.value:
        layer["pool_size"] = _get(_create_network_key(layer_index, "pool_size"))
        layer["stride"] = _get(_create_network_key(layer_index, "stride"))
        layer["padding"] = _get(_create_network_key(layer_index, "padding"))
    elif layer_type == GraphKeys.LayerTypes.BIRNN.value:
        layer["num_hidden"] = _get(_create_network_key(layer_index, "num_hidden"))
        layer["cell_type"] = _get(_create_network_key(layer_index, "cell_type"))
        layer["activation"] = _get(_create_network_key(layer_index, "activation"))
    elif layer_type == GraphKeys.LayerTypes.MDRNN.value:
        layer["num_hidden"] = _get(_create_network_key(layer_index, "num_hidden"))
        layer["kernel_size"] = _get(_create_network_key(layer_index, "kernel_size"))
        layer["cell_type"] = _get(_create_network_key(layer_index, "cell_type"))
        layer["activation"] = _get(_create_network_key(layer_index, "activation"))
    elif layer_type == GraphKeys.LayerTypes.DROPOUT.value:
        layer["keep_prob"] = _get(_create_network_key(layer_index, "keep_prob"))
    elif layer_type == GraphKeys.LayerTypes.L2_NORMALIZE:
        layer["axis"] = _get(_create_network_key(layer_index, "axis"))
    return layer


def _create_network_key(layer_index, param):
    network_key = "network[" + str(layer_index) + "][" + param + "]"
    return network_key


def generate_model_dict():
    resulting_dict = OrderedDict()
    network = []
    number_of_layers = len([value for key, value in request.form.items() if 'layer_type' in key.lower()])
    for layer_index in range(number_of_layers):
        layer = _get_layer(layer_index)
        network.append(layer)
    resulting_dict['network'] = network
    resulting_dict['output_layer'] = _get("output_layer")
    return resulting_dict


def save_model_as_json(directory, contents):
    model_name = _get('architecture_name') + ".json"
    with open(os.path.join(directory, model_name), 'w') as fp:
        json.dump(contents, fp, indent=4)


def _get(param):
    return request.form[param]


def _get_list(param):
    return request.form.getlist(param)


def start_training():
    command = [
        'python', 'train.py',
        '--architecture', create_path(app.config['ARCHITECTURES_DIRECTORY'],
                                      _get('architecture_name') + '.json'),
        '--dataset_dir', create_path(app.config['DATASET_DIRECTORY'],
                                     _get('dataset_name')),
        '--desired_image_size', _get('desired_image_size'),
        '--num_epochs', _get('num_epochs'),
        '--batch_size', _get('batch_size'),
        '--learning_rate', _get('learning_rate'),
        '--optimizer', _get('optimizer'),
        '--loss', _get('loss'),
        '--metrics', _get_list('metrics'),
    ]
    training_process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    for line in iter(training_process.stderr.readline, b''):
        time.sleep(1)
        yield line.rstrip().decode('utf-8')

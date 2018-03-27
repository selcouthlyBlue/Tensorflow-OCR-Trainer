import os
import json

from werkzeug.utils import secure_filename
from flask import request
from collections import OrderedDict

from trainer.backend import GraphKeys


def get_directory_list(path):
    directory_names = os.listdir(path)
    return directory_names

def upload_dataset(images, labels_file, dataset_directory, dataset_name):
    dataset_path = create_path(dataset_directory, dataset_name)
    os.makedirs(dataset_path)
    labels_file.save(create_path(dataset_path,
                                 secure_filename(labels_file.filename)))
    for image in images:
        image.save(create_path(dataset_path,
                               secure_filename(image.filename)))


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
    resulting_dict['output_layer'] = get("output_layer")
    return resulting_dict


def save_model_as_json(directory, model_name, contents):
    model_name = model_name + ".json"
    with open(os.path.join(directory, model_name), 'w') as fp:
        json.dump(contents, fp, indent=4)


def get(param):
    return request.form[param]


def get_list(param):
    return request.form.getlist(param)

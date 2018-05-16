import os
import json
import csv
import shutil
import multiprocessing
import time
import zipfile
import requests

from werkzeug.utils import secure_filename
from flask import request
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from trainer import app
from trainer.backend import GraphKeys, dataset_utils
from trainer.backend.dataset_utils import read_dataset_list
from trainer.backend.train_ocr import train_model
from trainer.backend.train_ocr import evaluate_model
from trainer.backend.train_ocr import continue_training_model
from trainer.backend import create_serving_model
from trainer.backend import visualize
from trainer.backend import create_optimized_graph


def _allowed_labels_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() \
           in app.config['ALLOWED_LABELS_FILE_EXTENSIONS']


def _allowed_image_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() \
           in app.config['ALLOWED_IMAGE_EXTENSIONS']


def _allowed_zip_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() \
           in app.config['ALLOWED_ZIP_EXTENSIONS']


def upload_dataset(dataset_zip):
    dataset_name = get('dataset_name')
    if _allowed_zip_file(dataset_zip.filename):
        dataset_zip_path = _create_path(app.config['DATASET_DIRECTORY'], secure_filename(dataset_zip.filename))
        dataset_zip.save(dataset_zip_path)
        dataset_path = _create_path(app.config['DATASET_DIRECTORY'], dataset_name)
        os.makedirs(dataset_path)
        _extract_zip_files(dataset_zip_path, dataset_path)
        delete_file(dataset_zip_path)
        split_dataset("labels.txt")
        return dataset_name + " has been uploaded."
    return "An error occurred in uploading the dataset."


def _extract_zip_files(src, dest_dir):
    zip_ref = zipfile.ZipFile(src, 'r')
    zip_ref.extractall(dest_dir)
    zip_ref.close()


def get_dataset(dataset_name):
    return _create_path(app.config['DATASET_DIRECTORY'], dataset_name)


def _get_number_of_lines(filename):
    counter = 0
    with open(filename) as f:
        for i, l in enumerate(f):
            counter = i
    return counter + 1


def get_dataset_list_with_amount_of_training_and_testing_data():
    dataset_list = []
    dataset_names = get_directory_list_from_config('DATASET_DIRECTORY')
    for dataset_name in dataset_names:
        dataset_dict = OrderedDict()
        dataset_path = get_dataset(dataset_name)
        number_training_samples = _get_number_of_lines(_create_path(dataset_path, 'train.csv'))
        number_testing_samples = _get_number_of_lines(_create_path(dataset_path, 'test.csv'))
        dataset_dict['name'] = dataset_name
        dataset_dict['num_training_examples'] = number_training_samples
        dataset_dict['num_testing_examples'] = number_testing_samples
        dataset_list.append(dataset_dict)
    return dataset_list


def get_directory_list_from_config(directory_key):
    directory_names = os.listdir(app.config[directory_key])
    return directory_names


def _create_labels_file(filename, features, labels):
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n', delimiter=' ')
        writer.writerows(zip(features, labels))


def split_dataset(labels_file):
    dataset_path = _create_path(app.config['DATASET_DIRECTORY'], get('dataset_name'))
    features, labels = read_dataset_list(_create_path(dataset_path,
                                                      secure_filename(labels_file)))
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=float(get('test_size')))
    _create_labels_file(_create_path(dataset_path, 'train.csv'), x_train, y_train)
    _create_labels_file(_create_path(dataset_path, 'test.csv'), x_test, y_test)


def _create_path(*args):
    return os.path.join(*args)


def get_enum_values(enum_type):
    return [key.value.replace('"', '') for key in enum_type]


def _get_layer(layer_index):
    layer = OrderedDict()
    layer["layer_type"] = get(_create_network_key(layer_index, "layer_type"))
    layer_type = layer["layer_type"]
    if layer_type == GraphKeys.LayerTypes.CONV2D.value:
        layer["num_filters"] = int(get(_create_network_key(layer_index, "num_filters")))
        layer["kernel_size"] = int(get(_create_network_key(layer_index, "kernel_size1")))
        if get(_create_network_key(layer_index, "kernel_size2")):
            layer["kernel_size"] = [layer["kernel_size"], int(get(_create_network_key(layer_index, "kernel_size2")))]
        layer["stride"] = int(get(_create_network_key(layer_index, "stride")))
        layer["padding"] = get(_create_network_key(layer_index, "padding"))
        layer["activation"] = get(_create_network_key(layer_index, "activation"))
    elif layer_type == GraphKeys.LayerTypes.MAX_POOL2D.value:
        layer["pool_size"] = int(get(_create_network_key(layer_index, "pool_size")))
        layer["stride"] = int(get(_create_network_key(layer_index, "stride")))
        layer["padding"] = get(_create_network_key(layer_index, "padding"))
    elif layer_type == GraphKeys.LayerTypes.BIRNN.value:
        layer["num_hidden"] = int(get(_create_network_key(layer_index, "num_hidden")))
        layer["cell_type"] = get(_create_network_key(layer_index, "cell_type"))
        layer["activation"] = get(_create_network_key(layer_index, "activation"))
    elif layer_type == GraphKeys.LayerTypes.MDRNN.value:
        layer["num_hidden"] = int(get(_create_network_key(layer_index, "num_hidden")))
        layer["cell_type"] = get(_create_network_key(layer_index, "cell_type"))
        layer["activation"] = get(_create_network_key(layer_index, "activation"))
    elif layer_type == GraphKeys.LayerTypes.DROPOUT.value:
        layer["keep_prob"] = float(get(_create_network_key(layer_index, "keep_prob")))
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
    return _create_path(app.config['MODELS_DIRECTORY'], model)


def visualize_model(model_name, host):
    model_path = get_model_path(model_name)
    visualization_task = multiprocessing.Process(target=visualize, args=(model_path, host))
    visualization_task.name = "visualize-{}".format(model_name)
    visualization_task.start()


def get_log(model_name, log_name):
    model_path = get_model_path(model_name)
    log_path = _create_path(model_path, log_name+".log")
    content = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
    return content


def request_connection(url):
    status_code = 404
    while status_code == 404:
        try:
            status_code = requests.get(url).status_code
        except requests.exceptions.ConnectionError:
            time.sleep(5)
            continue

def save_model_as_json():
    architecture_dict = _generate_architecture_dict()
    architecture_name = get('architecture_name')
    architecture_path = get_architecture_path(architecture_name)
    _write_json(architecture_path, architecture_dict)
    return architecture_name + " has been created."


def _write_json(path, content):
    with open(path, 'w') as fp:
        json.dump(content, fp, indent=0)


def get_architecture_path(architecture_name):
    return _create_path(app.config['ARCHITECTURES_DIRECTORY'], architecture_name) + ".json"


def _get_abs_path(filename):
    return _create_path(os.getcwd(), filename, '')


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


def _export_serving_model(model_name, input_name="features", output_names="output"):
    model_path = get_model_path(model_name)
    run_params = json.load(open(_create_path(model_path, "run_config.json")), object_pairs_hook=OrderedDict)
    serving_model_path, input_shape = create_serving_model(model_path,
                                                           run_params,
                                                           input_name)
    optimized_model_path = _create_path(model_path, app.config["OUTPUT_GRAPH_FILENAME"])
    create_optimized_graph(serving_model_path, output_names, output_graph_filename=optimized_model_path)
    delete_folder(serving_model_path)
    serving_model_config = OrderedDict()
    input_node = OrderedDict()
    image_config = OrderedDict()
    image_config['image_width'] = run_params['desired_image_width']
    image_config['image_height'] = run_params['desired_image_height']
    input_node['input_name'] = input_name
    input_node['input_shape'] = input_shape
    serving_model_config['input_nodes'] = [input_node]
    serving_model_config['output_names'] = output_names.split(',')
    serving_model_config_path = _create_path(model_path, app.config['SERVING_MODEL_CONFIG_FILENAME'])
    image_config_path = _create_path(model_path, app.config['IMAGE_CONFIG_FILENAME'])
    _write_json(serving_model_config_path, serving_model_config)
    _write_json(image_config_path, image_config)


def package_model_files(model_name):
    _export_serving_model(model_name)
    model_path = get_model_path(model_name)
    with zipfile.ZipFile(_create_path(model_path, app.config['MODEL_ZIP_FILENAME']), mode='w') as f_out:
        for model_file in [app.config['SERVING_MODEL_CONFIG_FILENAME'],
                           app.config["OUTPUT_GRAPH_FILENAME"],
                           app.config['IMAGE_CONFIG_FILENAME']]:
            model_file_path = _create_path(model_path, model_file)
            f_out.write(model_file_path, arcname=model_file)
            delete_file(model_file_path)
    return _get_abs_path(model_path)


def _test_task(model_name):
    checkpoint_dir = get_model_path(model_name)
    run_params = json.load(open(_create_path(checkpoint_dir, "run_config.json")), object_pairs_hook=OrderedDict)
    dataset_dir = get_dataset(run_params['dataset_name'])
    charset_file = run_params['charset_file']
    testing_task = multiprocessing.Process(target=evaluate_model,
                                           args=(run_params,
                                                 dataset_dir,
                                                 charset_file,
                                                 checkpoint_dir,
                                                 ' '))
    testing_task.start()
    return testing_task


def _retrain_task(model_name):
    checkpoint_dir = get_model_path(model_name)
    run_params = json.load(open(_create_path(checkpoint_dir, "run_config.json")), object_pairs_hook=OrderedDict)
    run_params['learning_rate'] = float(get('learning_rate'))
    run_params['checkpoint_epochs'] = int(get('checkpoint_epochs'))
    run_params['num_epochs'] = int(get('num_epochs'))
    dataset_dir = get_dataset(run_params['dataset_name'])
    run_config_path = _create_path(checkpoint_dir, 'run_config.json')
    _write_json(run_config_path, run_params)
    continue_training_task = multiprocessing.Process(target=continue_training_model,
                                                     args=(run_params,
                                                           checkpoint_dir,
                                                           dataset_dir))
    continue_training_task.start()
    return continue_training_task


def run_learning_task(task):
    if task == 'training':
        dataset_name = get('dataset_name')
        checkpoint_dir = get_model_path("model-" + time.strftime("%Y%m%d-%H%M%S"))
        running_task = _train_task(get('architecture_name'),
                                   dataset_name,
                                   checkpoint_dir,
                                   int(get('desired_image_width')),
                                   int(get('desired_image_height')),
                                   int(get('num_epochs')),
                                   int(get('checkpoint_epochs')),
                                   int(get('batch_size')),
                                   'charsets/chars.txt',
                                   float(get('learning_rate')),
                                   get('optimizer'),
                                   getlist('metrics'),
                                   get('loss'),
                                   get('validation_size'))
        _set_running_task_name(running_task, task, checkpoint_dir)
    elif task == 'testing':
        running_task = _test_task(get('model_name'))
        _set_running_task_name(running_task, task, get('model_name'))
    elif task == 'retrain':
        running_task = _retrain_task(get('model_name'))
        _set_running_task_name(running_task, task, get('model_name'))


def _set_running_task_name(running_task, task, checkpoint_dir):
    running_task.name = "{}-{}".format(task, checkpoint_dir)


def _train_task(architecture_name,
                dataset_name,
                checkpoint_dir,
                desired_image_width,
                desired_image_height,
                num_epochs,
                checkpoint_epochs,
                batch_size,
                charset_file,
                learning_rate,
                optimizer,
                metrics,
                loss,
                validation_size):
    if validation_size:
        validation_size = float(validation_size)
    dataset_dir = get_dataset(dataset_name)
    os.mkdir(checkpoint_dir)
    run_params = get_architecture_file_contents(architecture_name)
    classes = dataset_utils.get_characters_from(charset_file)
    run_params['loss'] = loss
    run_params['metrics'] = metrics
    run_params['desired_image_width'] = desired_image_width
    run_params['desired_image_height'] = desired_image_height
    run_params['batch_size'] = batch_size
    run_params['dataset_name'] = dataset_name
    run_params['charset_file'] = charset_file
    run_params['num_classes'] = len(classes) + 1
    run_params['checkpoint_epochs'] = checkpoint_epochs
    run_params['validation_size'] = validation_size
    run_params['num_epochs'] = num_epochs
    run_params['learning_rate'] = learning_rate
    run_params['optimizer'] = optimizer
    run_config_path = _create_path(checkpoint_dir, 'run_config.json')
    _write_json(run_config_path, run_params)
    task = multiprocessing.Process(target=train_model,
                                   args=(
                                       run_params,
                                       dataset_dir,
                                       checkpoint_dir,
                                       learning_rate,
                                       metrics,
                                       loss,
                                       optimizer,
                                       charset_file,
                                       validation_size,
                                       ' ',
                                       num_epochs,
                                       batch_size,
                                       checkpoint_epochs
                                   ))
    task.start()
    return task

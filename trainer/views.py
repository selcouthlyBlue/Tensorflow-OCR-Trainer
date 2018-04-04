import multiprocessing
from flask import request, render_template, flash, redirect, url_for

from trainer import app
from trainer.backend import GraphKeys
from trainer.controllers import create_path
from trainer.controllers import delete_architecture
from trainer.controllers import delete_dataset_folder
from trainer.controllers import get
from trainer.controllers import get_directory_list
from trainer.controllers import get_enum_values
from trainer.controllers import getlist
from trainer.controllers import generate_model_dict
from trainer.controllers import save_model_as_json
from trainer.controllers import split_dataset
from trainer.controllers import train_task
from trainer.controllers import upload_dataset


def _allowed_labels_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() \
           in app.config['ALLOWED_LABELS_FILE_EXTENSIONS']


def _allowed_image_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() \
           in app.config['ALLOWED_IMAGE_EXTENSIONS']


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/architectures', methods=['GET', 'POST'])
def architectures():
    if request.method == 'POST':
        resulting_dict = generate_model_dict()
        save_model_as_json(app.config['ARCHITECTURES_DIRECTORY'], resulting_dict)
    network_architectures = _get_network_architectures()
    return render_template("architectures.html",
                           network_architectures=network_architectures)


def _get_network_architectures():
    network_architectures = get_directory_list(app.config['ARCHITECTURES_DIRECTORY'])
    network_architectures = [network_architecture.split('.')[0]
                             for network_architecture in network_architectures]
    return network_architectures


@app.route('/create_network_architecture')
def create_network_architecture():
    return render_template("create_network_architecture.html",
                           layer_types=get_enum_values(GraphKeys.LayerTypes),
                           padding_types=get_enum_values(GraphKeys.PaddingTypes),
                           cell_types=get_enum_values(GraphKeys.CellTypes),
                           activation_functions=get_enum_values(GraphKeys.ActivationFunctions),
                           output_layers=get_enum_values(GraphKeys.OutputLayers))


@app.route('/delete/<architecture>', methods=['POST'])
def delete_architecture(architecture):
    delete_architecture(create_path(app.config['ARCHITECTURES_DIRECTORY'], architecture) + ".json")
    flash(architecture + " architecture deleted.")
    return redirect(url_for('architectures'))


@app.route('/dataset_form')
def dataset_form():
    return render_template("dataset_form.html")


@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
        if 'labels_file' not in request.files:
            flash('No labels file part.')
            return redirect(request.url)
        if 'images' not in request.files:
            flash('No images file part.')
            return redirect(request.url)
        labels_file = request.files['labels_file']
        if labels_file.filename == '':
            flash('No selected labels file')
            return redirect(request.url)
        images = request.files.getlist('images')
        if not images:
            flash('No images selected')
            return redirect(request.url)
        if (labels_file and _allowed_labels_file(labels_file.filename)) \
                and (images and _allowed_image_files(image.filename)
                     for image in images):
            upload_dataset(images, labels_file, app.config['DATASET_DIRECTORY'])
            split_dataset(labels_file, app.config['DATASET_DIRECTORY'])
    dataset_list = _get_dataset_list()
    return render_template("dataset.html", dataset_list=dataset_list)


@app.route('/delete_dataset/<dataset_name>', methods=['POST'])
def delete_dataset(dataset_name):
    delete_dataset_folder(create_path(app.config['DATASET_DIRECTORY'], dataset_name))
    return redirect(url_for('dataset'))


def _get_dataset_list():
    dataset_list = get_directory_list(app.config['DATASET_DIRECTORY'])
    return dataset_list


@app.route('/train')
def train():
    dataset_list = _get_dataset_list()
    network_architectures = _get_network_architectures()
    return render_template("train.html",
                           dataset_list=dataset_list,
                           network_architectures=network_architectures,
                           losses=get_enum_values(GraphKeys.Losses),
                           optimizers=get_enum_values(GraphKeys.Optimizers),
                           metrics=get_enum_values(GraphKeys.Metrics))


@app.route('/tasks/<task>', methods=['GET', 'POST'])
def tasks(task):
    running_tasks = []
    running_tasks.extend([running_task.name for running_task in multiprocessing.active_children()])
    if request.method == 'POST':
        if task == 'training':
            training_task = train_task(create_path(app.config['ARCHITECTURES_DIRECTORY'],
                                                   get('architecture_name') + '.json'),
                                       create_path(app.config['DATASET_DIRECTORY'],
                                                   get('dataset_name')),
                                       int(get('desired_image_size')),
                                       int(get('num_epochs')),
                                       int(get('checkpoint_epochs')),
                                       int(get('batch_size')),
                                       'charsets/chars.txt',
                                       float(get('learning_rate')),
                                       get('optimizer'),
                                       getlist('metrics'),
                                       get('loss'))
            training_task.name = task
            running_tasks.append(training_task.name)
        flash(task + " has started.")
    return render_template('tasks.html', running_tasks=running_tasks)


@app.route('/terminate/<task>', methods=['POST'])
def terminate(task):
    was_terminated_manually = False
    for running_task in multiprocessing.active_children():
        if running_task.name == task:
            running_task.terminate()
            running_task.join()
            was_terminated_manually = True
            flash(running_task.name.capitalize() + " is terminated.")
            break
    if not was_terminated_manually:
        flash(task.capitalize() + " was already terminated.")
    return redirect(url_for('tasks', task='view'))


def _render_progress(template_name, task):
    return render_template(template_name, task=task)


@app.errorhandler(404)
def url_error(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

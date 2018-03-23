from flask import request, render_template, flash, redirect

from trainer import app
from trainer.backend import GraphKeys
from trainer.controllers import upload_dataset
from trainer.controllers import get_directory_list
from trainer.controllers import get_enum_values
from trainer.controllers import generate_model_dict
from trainer.controllers import save_model_as_json

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
        save_model_as_json(app.config['ARCHITECTURES_DIRECTORY'], request.form['architecture_name'], resulting_dict)
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


@app.route('/upload_dataset')
def upload_dataset():
    return render_template("upload_dataset.html")


@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
        if 'labels_file' not in request.files:
            flash('No labels file part.')
            return redirect(request.url)
        if 'images[]' not in request.files:
            flash('No images file part.')
            return redirect(request.url)
        labels_file = request.files['labels_file']
        if labels_file.filename == '':
            flash('No selected labels file')
            return redirect(request.url)
        images = request.files.getlist("images[]")
        if not images:
            flash('No images selected')
            return redirect(request.url)
        if (labels_file and _allowed_labels_file(labels_file.filename)) \
                and (images and _allowed_image_files(image.filename)
                     for image in images):
            upload_dataset(images, labels_file,
                           app.config['DATASET_DIRECTORY'],
                           request.form['dataset_name'])
            flash(request.form['dataset_name'], " uploaded.")
    dataset_list = _get_dataset_list()
    return render_template("dataset.html", dataset_list=dataset_list)


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


@app.route('/training_progress', methods=['POST'])
def training_progress():
    pass


@app.errorhandler(404)
def url_error(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

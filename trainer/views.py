from flask import request, render_template, flash, redirect

from trainer import app
from trainer.controllers import *


def _allowed_labels_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_LABELS_FILE_EXTENSIONS']


def _allowed_image_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_IMAGE_EXTENSIONS']


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/create_network_architecture')
def create_network_architecture():
    return render_template("create_network_architecture.html")


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
                and (images and _allowed_image_files(image.filename) for image in images):
            upload_dataset(images, labels_file, app.config['DATASET_DIRECTORY'],
                           request.form['dataset_name'])
            flash(request.form['dataset_name'], " uploaded.")
    dataset_list = get_directory_list(app.config['DATASET_DIRECTORY'])
    return render_template("dataset.html", dataset_list=dataset_list)


@app.route('/train', methods=['POST'])
def train():
    train_params = request.form
    start_training(train_params)


@app.errorhandler(404)
def url_error(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

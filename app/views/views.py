from flask import render_template, request

from app import app
from architecture_enum import Architectures
from optimizer_enum import Optimizers
from train_using_tf_estimator import train

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/prepare_for_training')
def prepare_for_training():
    return render_template("train_prep.html",
                           architectures=Architectures,
                           optimizers=Optimizers
                           )

def _get(param, param_type):
    return request.form.get(param, type=param_type)

@app.route('/train', methods=['POST'])
def begin_training():
    labels_path = request.form['labels_path']
    dataset_path = request.form['dataset_path']
    desired_image_size = (_get('image_width', int), _get('image_height', int))
    architecture = Architectures(request.form['architecture'])
    num_hidden_units = _get('num_hidden_units', int)
    optimizer = Optimizers(request.form['solver_type'])
    learning_rate = _get('learning_rate', float)
    test_fraction = _get('validation_set_size', float)
    validation_steps = _get('validation_steps', int)
    num_epochs = _get('num_epochs', int)
    batch_size = _get('batch_size', int)

    train(
        labels_file=labels_path,
        data_dir=dataset_path,
        desired_image_size=desired_image_size,
        test_fraction=test_fraction,
        architecture=architecture,
        num_hidden_units=num_hidden_units,
        optimizer=optimizer,
        learning_rate=learning_rate,
        validation_steps=validation_steps,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    return render_template("begin_training.html")

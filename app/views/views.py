from flask import render_template, request
from app import app

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/prepare_for_training')
def prepare_for_training():
    return render_template("train_prep.html")

def get(param, param_type):
    return request.form.get(param, type=param_type)

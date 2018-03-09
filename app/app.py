from flask import Flask, request, render_template
from train_ocr import train

app = Flask(__name__)
app.config.from_object('config.DevelopmentConfig')

@app.route('/')
def index():
    return render_template("index.html")

@app.errorhandler(404)
def url_error(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

@app.route('/train', methods=['POST'])
def train():
    train_params = request.form
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


if __name__ == '__main__':
    app.run()

from flask import Flask
app = Flask(__name__)
app.secret_key = 'pancake'
app.config.from_object('config.DevelopmentConfig')

import trainer.views

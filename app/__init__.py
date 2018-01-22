from flask import Flask
from flask_bootstrap import Bootstrap

app = Flask(__name__, instance_relative_config=True)

from app.views import views

app.config.from_object('config')

import flask
import os

app = flask.Flask(__name__, static_folder = "static") 	# initialize a flask object

from RPS_webcam import routes

app.config["MODEL_FOLDER"] = os.path.abspath("RPS_webcam/static/models")






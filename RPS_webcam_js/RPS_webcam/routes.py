from RPS_webcam import app
from flask import render_template, send_from_directory



@app.route("/", methods=["GET", "POST"]) # tells Flask that this function is a URL endpoint and that data is being served from 'http://your_ip_address/'
def index():
	return render_template("index.html")


@app.route("/video_feed")
def video_feed():
	return render_template("video_feed.html")


@app.route("/models/<filename>")
def download_file(filename):
	return send_from_directory(app.config["MODEL_FOLDER"],filename = filename, as_attachment = True)

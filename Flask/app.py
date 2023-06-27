from __future__ import division, print_function
import os
import numpy as np
from keras.utils import img_to_array, load_img
from keras.models import load_model
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

global graph
graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)

model = load_model("breastcancer.h5")


@app.route("/", methods=["GET"])
def index():
    return render_template("bcancer.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)
        img = load_img(file_path, target_size=(64, 64))

        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        predictions = model.predict(images, batch_size=1)

        class_labels = np.argmax(predictions, axis=1)

        if class_labels[0] == 0:
            text = "The tumor is malignant.. check with a doctor"
        else:
            text = "The tumor is benign.. Need not worry"
        return text
    return ""


if __name__ == "__main__":
    app.run(debug=True, threaded=False)

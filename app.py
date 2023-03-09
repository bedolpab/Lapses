from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

generator = tf.keras.models.load_model("./dcgan-2/generator/")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict')
def predict():
    z = tf.random.normal([1, 128])
    image = generator.predict(z)[0, :, :, :]
    image = (1/(2*2.25)) * image + 0.5
    path = 'prediction.png'
    plt.imsave(path, image)
    return render_template('index.html')


if __name__ == "__main__":
    app.run()

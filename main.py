import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from skimage import img_as_float
import requests
import shutil
from io import BytesI
from tensorflow import keras
import numpy as np
from keras.layers import (
    Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape)
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.layers import LeakyReLU, Input
from keras.optimizers import Adam
from keras.datasets import mnist

# Neuron Structure


class Neuron:
    def __init__(self, data, pos):
        self.data = data
        self.pos = pos
        self.dendrites = []

    def Axon(self):
        return self.data


'''# Fetch image
url = "https://thispersondoesnotexist.com/image"
image = requests.get(url).content

with open('imp.png', 'wb') as f:
    f.write(image)

# Read image
image = io.imread("./img.png")

# Neuron Structure
neuron_graph = []
neuron_cells = []
count = 0
for i in range(image.shape[0]):
    row = []
    neuron_rows = []
    for j in range(image.shape[1]):
        row.append(image[i, j, :])
        data = image[i, j, :]
        pos = (i, j)

        # Create neuron
        neuron_rows.append(Neuron(data=data, pos=pos))
    # Append pixel row to re-display image
    neuron_graph.append(row)

    # Append neuron_row to graph
    neuron_cells.append(neuron_rows)

for i in range(len(neuron_cells)):
    for j in range(len(neuron_cells[0])):
        validator = lambda index: (0 <= index < i) and (0 <= index < j)
            '''

def get_images():
    return 0

def generator():
    return 0
    
def discriminator():
    return 0

def noise():
    return 0

def gan_network():
    return 0

def train():
    return 0

def sample_images():
    return 0


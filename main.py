import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from skimage import img_as_float
import requests
import shutil
from io import BytesIO
from PIL import Image

class Neuron:
    def __init__(self, data, neuron):
        self.data = data
        self.order = neuron

    def __str__(self):
        return f'Neuron {this.order} - Memory {this.data}'

url = "https://thispersondoesnotexist.com/image"
image = requests.get(url).content

with open('img.png', 'wb') as f:
    f.write(image)

image = io.imread("./img.png")
plt.imshow(image)

pixel_list = []
count = 0
for i in range(1023):
    for j in range(1023):
        pixel_list.append(image[i, j, :])
        count += 1

print(len(pixel_list))




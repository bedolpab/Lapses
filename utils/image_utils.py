import tensorflow as tf
from tensorflow import keras
from skimage import io
from skimage.io import imread_collection
from PIL import Image
from image_utils import make_dir
import matplotlib.pyplot as plt
import numpy as np
import requests
import shutil
import PIL
import cv2
import os

image_path = "./data/training"
current_img_type = "png"


def save_image(image, count):
    with open(f'{image_path}/img_{count}.{current_img_type}', 'wb') as f:
        f.write(image)


def fetch_images(k: int):
    """ 
    Fetch images from thispersondoesnotexist.com

    :param k: number of images to fetch
    :param folder_name: name of folder to save images to
    """
    if k < 1:
        return 0

    # Locals
    count = 0
    endpoint = 'image'
    url = f'https://thispersondoesnotexist.com/{endpoint}'
    while count < k:
        image = requests.get(url).content
        save_image(image, count)
        count += 1

        # A time.sleep(x) is recommended to avoid latency errors


def read_image(folder_name: str, image_name: str, img_type: str) -> np.ndarray:
    return io.imread(f'{folder_name}/{image_name}.{img_type}')


def image_exists(folder_name: str) -> bool:
    """ 
    Check whether an image exists in folder_name

    :param folder_name: folder in which dataset images are located
    """
    try:
        # Default image 0
        image = read_image(image_path, 'img_0', 0)
        return True
    except:
        print(f'Image "img_0.{current_img_type}" in {folder_name} not found')
        return False


def read_collection(folder_name: str) -> io.collection.ImageCollection:
    return imread_collection(f'./{folder_name}/*.{current_img_type}')


def resize_image(image, size) -> PIL.Image.Image:
    resized_image = cv2.resize(
        image, dsize=size, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(resized_image)


def resize_collection(folder_name: str, file_name: str, collection):
    make_dir(folder_name)

    for i in range(len(collection)):
        new_image = resize_image(collection[i], (128, 128))
        new_image.save(f'./{file_name}/img_{i}.png')

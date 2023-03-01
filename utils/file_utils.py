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


def make_directory(folder_name: str):
    if not os.path.exists(f'./{folder_name}'):
        os.makedirs(folder_name)

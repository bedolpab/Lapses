from skimage import io
from skimage.io import imread_collection
from PIL import Image
from utils.file_utils import make_directory
import numpy as np
import requests
import PIL
import cv2


image_path = "./data/training"


def save_image(image, count: int, img_type: str):
    with open(f'{image_path}/img_{count}.{img_type}', 'wb') as f:
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


def image_exists(folder_name: str, img_type: str) -> bool:
    """ 
    Check whether an image exists in folder_name

    :param folder_name: folder in which dataset images are located
    """
    try:
        # Default image 0
        image = read_image(image_path, 'img_0', 0)
        return True
    except:
        print(f'Image "img_0.{img_type}" in {folder_name} not found')
        return False


def read_collection(folder_name: str, img_type: str) -> io.collection.ImageCollection:
    return imread_collection(f'{folder_name}/*.{img_type}')


def resize_image(image, size) -> PIL.Image.Image:
    resized_image = cv2.resize(
        image, dsize=size, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(resized_image)


def resize_collection(folder_name: str,
                      img_type: str,
                      width: int,
                      height: int,
                      collection: io.collection.ImageCollection):
    make_directory(folder_name)

    for i in range(len(collection)):
        new_image = resize_image(collection[i], (width, height))
        new_image.save(f'{folder_name}/img_{i}.{img_type}')

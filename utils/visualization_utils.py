import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model


def show_batch(batch):

    # Default batch_size of 3
    images = np.random.randint(low=0, high=len(
        batch), size=9)  # get random indices
    fig = plt.figure(figsize=(3, 3))
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    cnt = 0
    for i in range(3):
        for j in range(3):
            # get image from batch at index 'i'
            axs[i, j].imshow(batch[images[cnt]])
            cnt += 1
    plt.show()


def model_to_png(model, file_name: str):
    plot_model(model, to_file=f'{file_name}.png', show_shapes=True)

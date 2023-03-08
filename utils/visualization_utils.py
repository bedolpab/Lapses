from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


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


def save_image(model, time):
    noise = tf.random.normal([1, 128])
    result = model.predict(noise)
    result = (result+1) / 2
    plt.imshow(result[0, :, :, :])
    plt.title(f'generation-{time}')
    plt.axis('off')
    plt.savefig(f'simulation/generation-{time}')


def model_to_png(model, file_name: str):
    plot_model(model, to_file=f'{file_name}.png', show_shapes=True)


def save_plot(data: list, label: str, path: str, file_name: str):
    plt.plot(data, label=label)
    plt.savefig(f'{path}/{file_name}.png')

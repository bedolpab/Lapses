from tensorflow import keras
from keras.models import Sequential
from utils.image_utils import read_collection, image_path
from utils.file_utils import make_directory
from utils.benchmark_utils import time_stamp, get_time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def DCGAN(generator, discriminator) -> keras.models.Sequential:
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    return model


def train_dcgan(iterations: int,
                batch_size: int,
                sample_interval: int,
                folder_name: str,
                generator: keras.models.Sequential,
                discriminator: keras.models.Sequential,
                dcgan: keras.models.Sequential):
    data_images = read_collection(image_path)
    image_count = 0

    discriminator_losses = []
    gan_losses = []
    # Labels
    time_stamp("Generating labels ...", get_time())
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    time_stamp("Finishing ...", get_time())

    make_directory(f'../{folder_name}')
    make_directory(f'../{folder_name}/predictions')

    # Training
    for iteration in range(iterations):
        # time_stamp(f'Iteration {iteration} of {iterations}', get_time())

        # Collect batch
        random_indicies = np.random.choice(
            len(data_images), size=batch_size, replace=False)
        real_image_batch = np.array(
            [data_images[i] for i in random_indicies]) / 127.5 - 1.0  # rescale [-1,1]

        # Random batch of fake images
        z_fake = tf.random.normal([batch_size, 128])

        generated_images = generator.predict(z_fake)

        # Train Discriminator -> [Loss, Accuracy]
        discriminator_real_loss = discriminator.train_on_batch(
            real_image_batch, real_labels)
        discriminator_fake_loss = discriminator.train_on_batch(
            generated_images, fake_labels)

        # Get Discriminator loss and accuracy
        discriminator_loss, accuracy = 0.5 * \
            np.add(discriminator_real_loss, discriminator_fake_loss)

        # Train Generator
        z_fake = tf.random.normal([batch_size, 128])
        generated_images = generator.predict(z_fake)

        # Get Generator loss and accuracy
        gan_loss = dcgan.train_on_batch(z_fake, real_labels)

        discriminator_losses.append(discriminator_loss)
        gan_losses.append(gan_loss)
        # Progress output
        if (iteration + 1) % sample_interval == 0:
            print("Iteration %d [D loss: %f, acc.:%.2f%%] [G loss: %f]" % (
                iteration + 1, discriminator_loss, 100.0 * accuracy, gan_loss))

            # Generate random images
            z_generated = tf.random.normal([3*3, 128])
            generate_images = generator.predict(z_generated)
            generate_images = 0.5 * generate_images + 0.5

            # Plot
            fig = plt.figure(figsize=(3, 3))
            fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
            cnt = 0
            for i in range(3):
                for j in range(3):
                    # get image from batch at index 'i'
                    axs[i, j].imshow(generate_images[cnt])
                    cnt += 1
            plt.savefig(
                f'../{folder_name}/predictions/iteration-{image_count}.png')
            image_count += 1
            plt.show()
    generator.save(f'../{folder_name}/generator')
    discriminator.save(f'../{folder_name}/discriminator')
    dcgan.save(f'../{folder_name}/gan')

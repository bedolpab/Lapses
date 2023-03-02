from models.dcgan import DCGAN
from models.discriminator import create_discriminator
from models.generator import create_generator
from keras.optimizers import Adam
from utils.image_utils import read_collection
from utils.file_utils import make_directory
from utils.benchmark_utils import time_stamp, get_time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.visualization_utils import save_plot
import config

# Create models
generator = create_generator(config.Z_DIM)
discriminator = create_discriminator(config.IMG_SHAPE)

# Compile generator
generator.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0002, beta_1=0.4))

# Compile discriminator
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=0.0002, beta_1=0.4),
                      metrics=['accuracy'])

# Disable discriminator training
discriminator.trainable = False


# Create DCGAN
dcgan = DCGAN(generator, discriminator)
dcgan.compile(loss='binary_crossentropy',
              optimizer=Adam())

discriminator_losses = []
gan_losses = []

# Train DCGAN
data_images = read_collection(config.DATA_TRAINING_PATH, 'jpg')

time_stamp("Generating labels ...", get_time())
real_labels = np.ones((config.BATCH_SIZE, 1))
fake_labels = np.zeros((config.BATCH_SIZE, 1))
time_stamp("Finishing ...", get_time())

make_directory(config.MODEL_FOLDER_NAME)
make_directory(f'{config.MODEL_FOLDER_NAME}/predictions')

for iteration in range(config.ITERATIONS):
    # Collect batch

    random_indicies = np.random.choice(
        len(data_images),
        size=config.BATCH_SIZE,
        replace=False)

    real_image_batch = np.array([data_images[i] for i in random_indicies]) / 127.5 - 1.0  # rescale [-1,1]

    # Random batch of fake images
    z_fake = tf.random.normal([config.BATCH_SIZE, config.Z_DIM])
    generated_images = generator.predict(z_fake)

    # Train Discriminator -> [Loss, Accuracy]
    discriminator_real_loss = discriminator.train_on_batch(
        real_image_batch, real_labels)
    discriminator_fake_loss = discriminator.train_on_batch(
        generated_images, fake_labels)

    # Get Discriminator loss and accuracy
    discriminator_loss, accuracy = 0.5 * np.add(
        discriminator_real_loss, discriminator_fake_loss)

    # Train Generator
    z_fake = tf.random.normal([config.BATCH_SIZE, config.Z_DIM])
    generated_images = generator.predict(z_fake)

    # Get Generator loss and accuracy
    gan_loss = dcgan.train_on_batch(z_fake, real_labels)

    discriminator_losses.append(discriminator_loss)
    gan_losses.append(gan_loss)

    # Progress output
    if (iteration + 1) % config.SAMPLE_INTERVAL == 0:
        print("Iteration %d [D loss: %f, acc.:%.2f%%] [G loss: %f]" % (
            iteration + 1, discriminator_loss, 100.0 * accuracy, gan_loss))

        # Generate random images
        z_generated = tf.random.normal([3*3, config.Z_DIM])
        generated_images = generator.predict(z_generated)
        generated_images = 0.5 * generated_images + 0.5

        # Plot
        fig = plt.figure(figsize=(3, 3))
        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)

        cnt = 0
        for i in range(3):
            for j in range(3):
                # Get images from batch at index 'i'
                axs[i, j].imshow(
                    generated_images[cnt, :, :, :]
                )
                cnt += 1
        plt.savefig(
            f'{config.MODEL_FOLDER_NAME}/predictions/{iteration+1}.png'
        )

plt.clf()
save_plot(discriminator_losses, 'Discriminator Loss',
          config.MODEL_FOLDER_NAME, 'discriminator_loss')

plt.clf()
save_plot(gan_losses, 'GAN Loss', config.MODEL_FOLDER_NAME, 'gan_loss')

generator.save(f'{config.MODEL_FOLDER_NAME}/generator')
discriminator.save(f'{config.MODEL_FOLDER_NAME}/discriminator')
dcgan.save(f'{config.MODEL_FOLDER_NAME}/dcgan')
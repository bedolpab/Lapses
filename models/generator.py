import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Reshape, Conv2DTranspose, Activation, BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential
import matplotlib.pyplot as plt


def create_generator(z) -> keras.models.Sequential:
    model = Sequential()

    # Input Latent vector
    model.add(Dense(4*4*128, input_dim=z))
    model.add(Reshape((4, 4, 128)))

    # Conv
    model.add(Conv2DTranspose(128, kernel_size=4, strides=4, padding='valid'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(32, kernel_size=3, strides=4, padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(16, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))
    model.summary()

    return model


def test_generator(z_dim: int):
    gen = create_generator(z_dim)
    noise = tf.random.normal([1, z_dim])
    img = gen.predict(noise)
    img = tf.reshape(img, shape=(img.shape[1], img.shape[2], img.shape[3]))
    plt.imshow(img)
    plt.show()

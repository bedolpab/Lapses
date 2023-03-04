import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Reshape, Conv2DTranspose, Activation, BatchNormalization, UpSampling2D
from keras.layers import Dropout, Activation, LeakyReLU
from keras.models import Sequential
from keras.utils import normalize
from keras.initializers.initializers_v2 import RandomNormal
import matplotlib.pyplot as plt


def create_generator(z) -> keras.models.Sequential:

    model = Sequential()

    # Input Latent vector
    model.add(Dense(4*4*128, input_dim=z))
    model.add(Reshape((4, 4, 128)))

    model.add(Conv2DTranspose(filters=128, kernel_size=4,
              padding='valid', strides=4, bias_initializer=RandomNormal(0, 1)))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2DTranspose(filters=64, kernel_size=3,
              padding='same', bias_initializer=RandomNormal(0, 1)))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2DTranspose(filters=32, kernel_size=3,
              padding='same', strides=4, bias_initializer=RandomNormal(0, 1)))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2DTranspose(filters=16, kernel_size=3,
              padding='same', bias_initializer=RandomNormal(0, 1)))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2DTranspose(filters=3, kernel_size=3, padding='same',
              strides=2, bias_initializer=RandomNormal(0, 1)))

    model.add(Activation('tanh'))

    model.summary()
    return model


def test_generator(z_dim: int):
    gen = create_generator(z_dim)
    noise = tf.random.normal([1, z_dim])
    img = gen.predict(noise)
    img = tf.reshape(img, shape=(img.shape[1], img.shape[2], img.shape[3]))
    plt.imshow(0.5 * img + 0.5)
    plt.show()

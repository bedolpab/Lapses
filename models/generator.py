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
    model.add(Dense(8*8*1024, input_dim=z, activation='linear'))
    model.add(Reshape((8, 8, 1024)))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(filters=512, kernel_size=5, padding='same', strides=2,
                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(epsilon=0.00005, trainable=True))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(filters=256, kernel_size=5, padding='same', strides=2,
                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(epsilon=0.00005, trainable=True))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(filters=128, kernel_size=5, padding='same', strides=2,
                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(epsilon=0.00005, trainable=True))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(filters=64, kernel_size=5, padding='same', strides=2,
                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(epsilon=0.00005, trainable=True))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(filters=3, kernel_size=5, padding='same', strides=1,
                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
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

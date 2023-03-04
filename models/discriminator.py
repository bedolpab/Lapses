
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import LeakyReLU, BatchNormalization, AveragePooling2D
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.initializers.initializers_v2 import RandomNormal


def create_discriminator(img_shape) -> keras.models.Sequential:

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=5, padding='same', input_shape=img_shape,
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), strides=2))
    model.add(BatchNormalization(epsilon=0.00005, trainable=True))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(filters=128, kernel_size=5, padding='same', strides=2,
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(epsilon=0.00005, trainable=True))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(filters=256, kernel_size=5, padding='same', strides=2,
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(epsilon=0.00005, trainable=True))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(filters=512, kernel_size=5, padding='same', strides=1,
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(epsilon=0.00005, trainable=True))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(filters=1024, kernel_size=5, padding='same', strides=2,
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(epsilon=0.00005, trainable=True))
    model.add(LeakyReLU(alpha=0.2))

    # Finalized
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.add(Activation('sigmoid'))
    model.summary()

    return model

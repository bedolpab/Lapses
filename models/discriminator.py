
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import LeakyReLU, BatchNormalization, AveragePooling2D
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.initializers.initializers_v2 import RandomNormal


def create_discriminator(img_shape) -> keras.models.Sequential:

    model = Sequential()

    model.add(Conv2D(filters=3, kernel_size=1,
              padding='same', input_shape=img_shape))
    model.add(Conv2D(filters=3, kernel_size=3, padding='same'))

    model.add(Conv2D(filters=16, kernel_size=3, padding='same', strides=2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32, kernel_size=2, padding='valid', strides=2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=2, padding='valid', strides=2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=2, padding='valid', strides=2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=2, padding='valid', strides=2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=2, padding='valid', strides=2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=2, padding='valid', strides=2))
    model.add(LeakyReLU(alpha=0.2))

    # Finalized
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model

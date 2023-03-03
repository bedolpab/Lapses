
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import LeakyReLU, BatchNormalization, AveragePooling2D
from keras.models import Sequential
from keras.layers import Dropout, Activation


def create_discriminator(img_shape) -> keras.models.Sequential:

    model = Sequential()

    model.add(Conv2D(3, kernel_size=5, padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))

    model.add(Conv2D(16, kernel_size=5, padding='same', strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))

    model.add(Conv2D(32, kernel_size=5, padding='same', strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))

    model.add(Conv2D(64, kernel_size=5, padding='same', strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))

    model.add(Conv2D(64, kernel_size=5, padding='same', strides=2))
    model.add(Activation('relu'))

    # Finalized
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model

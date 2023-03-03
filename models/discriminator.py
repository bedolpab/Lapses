
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Dropout


def create_discriminator(img_shape) -> keras.models.Sequential:

    model = Sequential()
    model.add(Conv2D(3, kernel_size=1, padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(3, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(16, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(32, kernel_size=2, strides=2, padding='valid'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=2, strides=2, padding='valid'))
    model.add(LeakyReLU(alpha=0.2))

    # Finalized
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

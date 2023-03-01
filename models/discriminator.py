
from tensorflow import keras
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D
from keras.layers import LeakyReLU
from keras.models import Sequential


def create_discriminator(img_shape) -> keras.models.Sequential:
    model = Sequential()
    model.add(Conv2D(3, kernel_size=1, padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    # Conv 1.0
    model.add(Conv2D(16, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    # Conv 2.0
    model.add(Conv2D(32, kernel_size=2, strides=2, padding='valid'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    # Conv 3.0
    model.add(Conv2D(64, kernel_size=2, strides=2, padding='valid'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    # Conv 3.0
    model.add(Conv2D(128, kernel_size=2, strides=2, padding='valid'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    # Conv 4.0
    model.add(Conv2D(128, kernel_size=2, strides=2, padding='valid'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, kernel_size=2, strides=2, padding='valid'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    # Conv 5.0
    model.add(Conv2D(128, kernel_size=2, strides=2, padding='valid'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    # Finalized
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

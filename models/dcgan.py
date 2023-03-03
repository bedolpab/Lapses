from tensorflow import keras
from keras.models import Sequential


def DCGAN(generator, discriminator) -> keras.models.Sequential:
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    model.summary()
    return model

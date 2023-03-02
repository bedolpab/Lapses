from models.dcgan import DCGAN, train_dcgan
from models.discriminator import create_discriminator
from models.generator import create_generator
from keras.optimizers import Adam
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

# Train DCGAN
train_dcgan(config.ITERATIONS,
            config.BATCH_SIZE,
            config.SAMPLE_INTERVAL,
            config.MODEL_FOLDER_NAME,
            config.DATA_TRAINING_PATH,
            generator,
            discriminator,
            dcgan)

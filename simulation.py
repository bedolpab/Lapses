from keras.models import load_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.benchmark_utils import time_stamp, get_time
from utils.visualization_utils import save_image

# Assuming our "AI" is around the average age of the typicall average of someone diagnoseed
# with dementia, we will allow it a simulation of 9 years

# Assuming there are 365 days in a calendar year
# Leap years are 366, assuming only one year is allowed and the final one isn't complete

# Neuronal Death remains unknown,

# Our DCGAN models Alzheimers in particular for simplicfication purpouses, thus we randomize at which layer
# neurons are killed since our model merely represents the neocortex and hippocampus parts of the brain.
# In specific, we model the hippocampus part of the brain very abstractly. Thus, our generator models
# this version since its reponsible for memory (abtractly).

# That being said, we'll kill neurons at random at any layer in this portion of the brain.

# Valid layers to kill 0, 2 .., 10

regular_days = 365
leap_year_days = 366
years = 8
total_days = (years * regular_days) + leap_year_days
rate_nk = 0.99
rate_yk = 0.01
valid_layers = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def simulate(time: int):
    # Load model to apply simulation on
    model = load_model('./dcgan-2/generator')

    time_stamp("Begining Simulation", get_time())
    for i in range(time):

        if i > 0:
            model = load_model(f'simulation/sim-models/model-gen-{i-1}')

        # Get model weights & Biases
        weights = model.layers[0].get_weights()[0]
        biases = model.layers[0].get_weights()[1]

        # Get neuron mask (makes weights 0)
        mask = np.random.choice(
            [0, 1], size=weights.shape, p=[rate_yk, rate_nk])

        # Apply mask
        weights = weights * mask
        biases = biases * mask.mean(axis=0)

        # Set
        model.layers[0].set_weights([weights, biases])
        save_image(model, i)
        model.save(f'simulation/sim-models/model-gen-{i}')

    time_stamp("Ending Simulation", get_time())


simulate(total_days)

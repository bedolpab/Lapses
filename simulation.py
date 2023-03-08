from keras.models import load_model
from utils.benchmark_utils import time_stamp, get_time
from utils.visualization_utils import save_image
import numpy as np

regular_days = 365
leap_year_days = 366
years = 8
total_days = (years * regular_days) + leap_year_days
rate_nk = 0.99
rate_yk = 0.01


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

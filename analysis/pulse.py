import numpy as np
import matplotlib.pyplot as plt

# Samllest Time Step
TAU = 1e-3  # 1 ms
MAX = 0.25


def generate_pulse_vector(spikes, tau=TAU, duration=MAX):
    """Generates a pulse vector from a spike 3D array."""
    round_factor = len("{:.10f}".format(tau).split('.')[1])
    rounded_spikes = np.round(spikes, round_factor)

    pulse = np.zeros(int(duration / tau))

    non_zero_indices = np.argwhere(rounded_spikes != 0)

    for i, j, k in non_zero_indices:
        index = int(rounded_spikes[i, j, k] / tau)
        pulse[index] += 1

    return pulse


def generate_pulse_vector_opt(spikes, tau=TAU, duration=MAX):
    """
    Generates a pulse vector from a spike 3D array.
    Not as accurate but significantly faster.
    """
    round_factor = len("{:.10f}".format(tau).split('.')[1])
    rounded_spikes = np.round(spikes, round_factor)

    pulse = np.zeros(int(duration / tau))

    non_zero_spikes = rounded_spikes[rounded_spikes != 0]
    indices = (non_zero_spikes / tau).astype(int)

    np.add.at(pulse, indices, 1)

    return pulse


def euclidean_distance(u, v):
    """Compute the Euclidean distance between two vectors."""
    return np.linalg.norm(u - v)


def prediction_error(given, expected):
    """Compute the prediction error between two vectors."""
    return np.sum(given - expected)


def cosine_similarity(u, v):
    """Compute the cosine similarity between two vectors."""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def analyze(spikes_list, comparison_function):
    pulses = [generate_pulse_vector(spikes) for spikes in spikes_list]
    comps = [comparison_function(pulse, pulses[0]) for pulse in pulses]
    for comp in comps:
        print(comp)
    plt.plot(comps)

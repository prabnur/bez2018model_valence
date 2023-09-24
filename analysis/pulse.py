from model import get_spikes, normalize_to_range
import numpy as np
import matplotlib.pyplot as plt

# Samllest Time Step
TAU = 1e-3  # 0.001 ms
MAX = 0.25

def generate_pulse_vector(spikes):
    """Generates a pulse vector from a spike 3D array."""
    round_factor = len("{:.10f}".format(TAU).split('.')[1])
    rounded_spikes = np.round(spikes, round_factor)
    pulse = np.zeros(int(MAX / TAU))
    for cf in rounded_spikes:
        for anf in cf:
            for spike in anf:
                if spike != 0:
                    pulse[int(spike / TAU)] += 1
    return pulse


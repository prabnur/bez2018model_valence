import numpy as np
from analysis.spike_tensor import generate_spike_tensor
from cache import get_spikes


def generate_probabilities_simple(note, mockTensors=None):
    if mockTensors is not None:
        tensors = mockTensors
    else:
        notes_spikes = get_spikes(note, mode="rng")
        tensors = [generate_spike_tensor(spikes) for spikes in notes_spikes]

    mu = 1 / len(tensors)

    probability_tensor = np.zeros_like(tensors[0], dtype=float)
    for tensor in tensors:
        probability_tensor += tensor * mu

    return probability_tensor

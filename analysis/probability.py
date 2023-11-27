import numpy as np
from analysis.pulse import TAU
from analysis.spike_tensor import generate_spike_tensor
from cache import get_spikes


def get_tensors(note, tau=TAU, mockTensors=None):
    if mockTensors is not None:
        return mockTensors

    notes_spikes = get_spikes(note, mode="rng")
    return [generate_spike_tensor(spikes, tau=tau) for spikes in notes_spikes]


# CUMULATIVE

# DELTA = 0.01

# def cumulative_reduction(note, delta=DELTA, mockTensors=None):
#     tensors = get_tensors(note, mockTensors=mockTensors)

#     probability_tensor = np.zeros_like(tensors[0], dtype=float)
#     num_cf, num_anf, time_period = np.shape(tensors[0])

#     P_current = np.zeros((num_cf, num_anf), dtype=float)

#     for tensor in tensors:
#         for cf_idx, cf in enumerate(tensor):
#             for anf_idx, anf in enumerate(cf):
#                 for t in range(time_period):
#                     P_current[cf_idx][anf_idx] += delta if anf[t] == 1 else -delta
#                     probability_tensor[cf_idx, anf_idx, t] = P_current[cf_idx][anf_idx]

#     return probability_tensor


def cumulative_reduction_optimized(note, tau=0.001, mockTensors=None):
    tensors = get_tensors(note, tau=tau, mockTensors=mockTensors)

    num_cf, num_anf, time_period = tensors[0].shape
    probability_tensor = np.zeros((num_cf, num_anf, time_period), dtype=float)
    P_current = np.zeros((num_cf, num_anf), dtype=float)

    delta = 1 / time_period

    for tensor in tensors:
        for t in range(time_period):
            delta_tensor = (tensor[:, :, t] == 1) * delta - (tensor[:, :, t] != 1) * delta
            P_current += delta_tensor
            probability_tensor[:, :, t] = P_current

    return probability_tensor


def cumulative_average(note, tau=0.001, mockTensors=None):
    tensors = get_tensors(note, tau=tau, mockTensors=mockTensors)
    tensors = np.array(tensors)

    time_period = tensors.shape[3]
    delta = 1 / time_period

    probability_tensors = np.cumsum(np.where(tensors == 1, delta, -delta), axis=3)

    return np.mean(probability_tensors, axis=0)


# SIMPLE

def generate_probabilities_simple(note, mockTensors=None):
    tensors = get_tensors(note, mockTensors=mockTensors)

    mu = 1 / len(tensors)

    probability_tensor = np.zeros_like(tensors[0], dtype=float)
    for tensor in tensors:
        probability_tensor += tensor * mu

    return probability_tensor


def simple_posneg(note, mockTensors=None):
    tensors = get_tensors(note, mockTensors=mockTensors)

    mu = 1 / len(tensors)
    probability_tensor = np.zeros_like(tensors[0], dtype=float)
    for tensor in tensors:
        probability_tensor += np.where(tensor == 1, mu, -mu)

    return probability_tensor

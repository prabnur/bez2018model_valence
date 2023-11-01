import numpy as np
from analysis.pulse import TAU, MAX

SNAP_SIZE = 5


def generate_spike_tensor(spikes, tau=TAU, duration=MAX):
    round_factor = len("{:.10f}".format(tau).split('.')[1])
    rounded_spikes = np.round(spikes, round_factor)

    num_cf, num_anf_per_cf, _ = np.shape(spikes)
    spike_tensor = np.zeros((num_cf, num_anf_per_cf, round(duration / tau)))

    spike_indices = np.round(rounded_spikes / tau) - 1
    spike_indices = spike_indices.astype(int)

    cf_indices, anf_indices, _ = np.indices(spikes.shape)

    valid_spikes = (spike_indices >= 0) & (spike_indices < round(duration / tau))

    spike_tensor[cf_indices[valid_spikes], anf_indices[valid_spikes], spike_indices[valid_spikes]] = 1

    return spike_tensor


def generate_snapshot(tensor, expectation, snap_size=SNAP_SIZE):
    snapshot = [[] for i in range(snap_size)]

    def add_to_snap(cf_idx, anf_idx, idx):
        start = idx - (snap_size // 2)
        for i in range(start, start + snap_size):  # Modify the range to start from `start`
            relative_idx = i - start
            if 0 <= relative_idx < snap_size and 0 <= i < len(expectation[cf_idx][anf_idx]):
                snapshot[relative_idx].append(expectation[cf_idx][anf_idx][i])

    for cf_idx, cf in enumerate(tensor):
        for anf_idx, anf in enumerate(cf):
            for spike_idx, spike in enumerate(anf):
                if spike == 1:
                    add_to_snap(cf_idx, anf_idx, spike_idx)

    snapshot = [(sum(snap) / len(snap) if len(snap) > 0 else 0) for snap in snapshot]
    return snapshot


def generate_expectation(tensors, scores):

    # We want floating point type so can't use zeros_like
    expectation = np.zeros(np.shape(tensors[0]))

    for tensor, score in zip(tensors, scores):
        expectation += tensor * score

    # expectation = expectation / len(tensors)
    return expectation

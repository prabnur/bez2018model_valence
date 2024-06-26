import numpy as np
from analysis.musical import CONS_RANK
from analysis.pulse import TAU, MAX


def generate_spike_tensor(spikes, tau=TAU, duration=MAX):
    round_factor = len("{:.10f}".format(tau).split('.')[1])
    rounded_spikes = np.round(spikes, round_factor)

    # duration = max(duration, np.max(rounded_spikes))

    num_cf, num_anf_per_cf, _ = np.shape(spikes)
    spike_tensor = np.zeros(
        (num_cf, num_anf_per_cf, round(duration / tau))
    )

    for cf_idx, cf in enumerate(rounded_spikes):
        for anf_idx, anf in enumerate(cf):
            for spike_time in anf:
                if spike_time == 0 or spike_time > duration:
                    continue
                spike_tensor[cf_idx][anf_idx][round(spike_time / tau) - 1] = 1

    return spike_tensor

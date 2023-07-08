from .bez2018model import nervegram
from .num_spike_trains import NUM_SPIKE_TRAINS_LIST
import numpy as np

def call_model(sound_data):
    np.random.seed(711)

    # get data
    input_signal_fs, input_signal = sound_data

    result = nervegram(
        input_signal,
        input_signal_fs,

        # Params worth changing
        synapseMode=1, # 0 = less computation
        max_spikes_per_train=200,
        nervegram_fs=20e3,

        # Params not worth changing
        species=2, # Human (Shera et al. 2002)
        num_cf=3500,
        min_cf=20,
        max_cf=16e3,
        num_spikes_per_train=NUM_SPIKE_TRAINS_LIST,
        implnt=1, # 0 = approximate, 1 = actual Power Law

        # What is returned
        return_vihcs=False,
        return_meanrates=False,
        return_spike_times=True,
        return_spike_tensor_sparse=False,
        return_spike_tensor_dense=False,
    )
    return result["nervegram_spike_times"]


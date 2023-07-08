from .bez2018model import nervegram
# import .cython_bez2018
import numpy as np

def call_model(sound_data):
    np.random.seed(711)

    # get data
    input_signal_fs, input_signal = sound_data
    num_spike_trains_list = generate_num_spike_trains(3500)

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
        num_spikes_per_train=num_spike_trains_list,
        implnt=1, # 0 = approximate, 1 = actual Power Law

        # What is returned
        return_vihcs=False,
        return_meanrates=False,
        return_spike_times=True,
        return_spike_tensor_sparse=False,
        return_spike_tensor_dense=False,
    )
    return result["nervegram_spike_times"]

def generate_num_spike_trains(num_cf):
    def generate_poisson(low, high, lam, size=1):
        while True:
            sample = np.random.poisson(lam, size)
            if low <= sample <= high:
                return sample[0]

    size = num_cf  # Number of samples you want
    low, high = 10, 20  # The range
    lam = 15  # lambda

    return [generate_poisson(low, high, lam) for _ in range(size)]

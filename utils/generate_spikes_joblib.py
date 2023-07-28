# Archived: Not working as expected

from .bez2018model import nervegram, get_ERB_cf_list
import numpy as np
from joblib import Parallel, delayed, wrap_non_picklable_objects
from tqdm import tqdm

def generate_spikes_joblib(sound_data, duration=0.25):
    np.random.seed(711)

    # get data
    input_signal_fs, input_signal = sound_data
    input_signal = input_signal[:int(duration*input_signal_fs)]

    cf_list = get_ERB_cf_list(num_cf=3500, min_cf=125, max_cf=16e3)
    sponts = np.load("./model/sponts.npy")
    Args = [
        (cf_list[i], sponts[i])
            for i in range(len(cf_list))
    ]
    
    # num_cpu = max(os.cpu_count(), cpu_count())
    @delayed
    @wrap_non_picklable_objects
    def generate(args):
        np.random.seed(711)
        cf, spont = args
        # print("Start")
        result = nervegram(
            input_signal,
            input_signal_fs,

            # Params worth changing
            max_spikes_per_train=200,
            nervegram_fs=20e3,

            # Params not worth changing
            synapseMode=1, # 0 = less computation
            species=2, # Human (Shera et al. 2002)
            cf_list=[cf],
            spont=spont,
            implnt=1, # 0 = approximate, 1 = actual Power Law
            num_spike_trains=1,

            # What is returned
            return_vihcs=False,
            return_meanrates=False,
            return_spike_times=True,
            return_spike_tensor_sparse=False,
            return_spike_tensor_dense=False,
        )
        # print("End")
        return result["nervegram_spike_times"][0][0]
        
    results = Parallel(n_jobs=-1)(generate(Args[i]) for i in tqdm(range(len(Args))))
    return np.array(results)
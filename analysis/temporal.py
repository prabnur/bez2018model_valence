from collections import defaultdict
import numpy as np

to_compare = [
    "C4",
    "C5",
    "G4",
    "F4",
    "E4",
    "A4",
    "D4",
    "B4",
    "Ab4",
    "Eb4",
    "Bb4",
    "Gb4",
    "Db4",
]


def calc_avg_isi(spike_times):
    """
    Helper function to get average InterSpike Interval for some spike times.
    eg
    [1.0,2.0,3.0,0.0,0.0,0.0,] -> 1.0
    """
    # remove all zeros in spike_times
    spike_times = spike_times[spike_times != 0]
    # Add 0 at beginning
    spike_times = np.insert(spike_times, 0, 0)
    # calculate differences between adjacent elements
    interspike_intervals = np.diff(spike_times)
    # calculate mean of interspike_intervals
    mean_isi = np.mean(interspike_intervals)
    # print(mean_isi)
    return mean_isi


def get_avg_isi(spikes):
    """
    Helper function to get average InterSpike Intervals for each spike trains for some note.
    eg
    shape(15, 3500, 200) -> [0.3, 0.4 ....] (len: 15*3500)
    """
    return np.apply_along_axis(calc_avg_isi, 2, spikes).flatten()

import numpy as np

CONCURRENCY_SCALE = 8
# root_spikes = np.around(root_spikes, decimals=CONCURRENCY_SCALE)

def create_concurrency_profile(CF, norm=True):
    # Initialize the concurrency profile with zeros
    concurrency_profile = [0]*18
    
    # Create a dictionary to store spike times and their occurrences
    spike_time_occurrences = defaultdict(int)
    
    # Flatten the spike times and count occurrences
    for anf in CF:
        for spike_time in anf:
            if spike_time != 0:  # Ignore zero padding
                spike_time_occurrences[round(spike_time, CONCURRENCY_SCALE)] += 1  # Rounding to 4 decimal places as mentioned
    
    # Count concurrent spikes for each unique spike time
    for spike_time, count in spike_time_occurrences.items():
        concurrency_profile[count - 1] += 1

    if norm:
        total = sum(concurrency_profile)
        concurrency_profile = [(count/total)*100 for count in concurrency_profile]

    return concurrency_profile
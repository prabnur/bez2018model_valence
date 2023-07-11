import numpy as np
from model import get_spikes

to_compare = ["C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4"]
c_major_scale = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]


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
    return np.apply_along_axis(calc_avg_isi, 2, spikes).T.flatten()

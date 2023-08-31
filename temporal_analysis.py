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

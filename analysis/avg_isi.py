from model import get_spikes
from analysis.temporal import get_avg_isi, to_compare
import matplotlib.pyplot as plt
import numpy as np

# root_note = "C4"
# root_spikes = get_spikes(root_note)
# root_avg_isi = get_avg_isi(root_spikes)

# avg_isi_s = [(note, get_avg_isi(get_spikes(note))) for note in to_compare]

def plot_avg_isi(note, avg_isi):
    x = list(range(len(avg_isi)))
    y = avg_isi
    plt.plot(x, y)
    plt.title(f"Average ISI for {note}")
    plt.xlabel("Spike Train")
    plt.ylabel("Average ISI")
    plt.show()

def plot_counts(note, counts):
    x = np.arange(len(counts))
    plt.plot(x, counts)
    plt.title(note)
    plt.xlabel("Spike Train")
    plt.ylabel("Counts")
    plt.show()
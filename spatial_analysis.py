import numpy as np
from model import get_spikes


def count_spikes(note):
    num_trains = len(note)
    num_cf = len(note[0])

    counts = []
    for cf_idx in range(num_cf):
        for train_idx in range(num_trains):
            nonzero = np.count_nonzero(note[train_idx][cf_idx])
            counts.append(nonzero)
    return counts


def count_spikes_optimized(note):
    return np.count_nonzero(note, axis=2).T.flatten()


def compare(note_a, note_b, scale=None):
    spikes_a = get_spikes(note_a)
    spikes_b = get_spikes(note_b)

    counts_a = count_spikes_optimized(spikes_a)
    counts_b = count_spikes_optimized(spikes_b)

    diff = counts_a - counts_b
    sum = np.sum(diff) / len(diff)
    abs_sum = np.sum(np.abs(diff)) / len(diff)
    if scale is not None and note_b in scale:
        print(f"Notes: {note_a} - {note_b}: SCALE")
    else:
        print(f"Notes: {note_a} - {note_b}")
    print(f"Abs Sum: {abs_sum} ; Sum: {sum}")


to_compare = ["C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4"]
c_major_scale = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]
for note in to_compare:
    compare("C4", note, c_major_scale)

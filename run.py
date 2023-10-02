import os
import numpy as np
from tqdm import tqdm
from analysis.spectral import decode
from model import SPIKES_DIR, first_n_primes, save_spikes_rng, save_spikes
from analysis.temporal import create_concurrency_profile

# Stereo
# (num_spike_trains, CFs, num_channels (2), spikes_per_train)
# eg. (40, 1, 2, 100)

# Mono
# (num_spike_trains, CFs, spikes_per_train)
# wg. (40, 1, 100)


def generate_decoded_exp():
    DECODED_DIR = "./decoded/exp/"
    directory = os.path.dirname(DECODED_DIR)

    # Create the directory recursively if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    for filename in tqdm(os.listdir(SPIKES_DIR)):
        if not filename.endswith(".npy"):
            continue

        # Generate
        spikes = np.load(os.path.join(SPIKES_DIR, filename))
        decoded = np.array([])

        for cf in spikes:
            conc_profile = create_concurrency_profile(cf)
            decoded_profile = decode(conc_profile)
            decoded = np.append(decoded, decoded_profile)

        name, _ = os.path.splitext(filename)
        # Save
        np.save(os.path.join(DECODED_DIR, name), decoded)


if __name__ == "__main__":
    # generate_scale(4)
    # save_spikes("C4")
    # save_spikes("C5", instrument="bells")
    # save_spikes("C5", instrument="flute")
    # save_spikes("C5", instrument="violin")
    notes = ["C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4", "C5"]
    seeds = first_n_primes(30)
    for seed in seeds:
        for note in notes:
            save_spikes_rng(note, seed)

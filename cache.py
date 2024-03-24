import numpy as np
import os
from scipy.io import wavfile

REPO_PATH = "/home/prab/Documents/bez2018model_valence"
CACHED = "cached"


def abspath(path):
    return os.path.join(REPO_PATH, CACHED, path)


SPIKES_DIR = abspath("spikes")
NOTES_DIR = "/Users/prabnurbal/Documents/notes"
RNG_SPIKES_DIR = abspath("spikes/piano/rng")
TONE_SPIKES_DIR = abspath("spikes/tone")
SNAPSHOT_DIR = abspath("snapshots/posneg")
EXPECTATION_DIR = abspath("expectations/piano")


def save(path, array):
    # Extract the directory name from the file path
    directory = os.path.dirname(path)

    # Create the directory recursively if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save the numpy array to the path
    np.save(path, array)


def sharp_to_flat(sharp_note):
    # Mapping of sharp notes to flat notes
    sharp_to_flat_map = {
        "C#": "Db",
        "D#": "Eb",
        "E#": "F",
        "F#": "Gb",
        "G#": "Ab",
        "A#": "Bb",
        "B#": "C",
    }
    # Return the flat equivalent if it exists, otherwise return the input note
    return sharp_to_flat_map.get(sharp_note, sharp_note)


def get_spikes(note, mode="regular", instrument="piano"):
    if mode == "regular":
        if note[1] == "#":
            note = sharp_to_flat(note[:2]) + note[2:]
        return np.load(os.path.join(os.path.join(SPIKES_DIR, instrument), f"{note}.npy"))
    elif mode == "rng":
        rng_path = os.path.join(SPIKES_DIR, instrument, "rng", note)
        return [
            np.load(os.path.join(rng_path, filename))
            for filename in os.listdir(rng_path)
            if filename.startswith(note)
        ]


def load_note_sound(note, instrument="piano"):
    if note[1] == "#":
        note = sharp_to_flat(note[:2]) + note[2:]

    return wavfile.read(os.path.join(NOTES_DIR, instrument, f"{note}.wav"))

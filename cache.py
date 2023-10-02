import numpy as np
import os
from scipy.io import wavfile


NOTES_DIR = "../notes/Piano"
SPIKES_DIR = "./spikes/piano/abs"
RNG_SPIKES_DIR = "./spikes/piano/rng"


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
    if note[1] == "#":
        note = sharp_to_flat(note[:2]) + note[2:]

    if mode == "regular":
        return np.load(os.path.join(f"./spikes/{instrument}", f"{note}.npy"))
    elif mode == "rng":
        final = []
        for filename in os.listdir(RNG_SPIKES_DIR):
            if filename.startswith(note):
                final.append(np.load(os.path.join(RNG_SPIKES_DIR, filename)))
        return final


def load_note_sound(note, instrument="piano"):
    notes_dir = f"../notes/{instrument}"

    if note[1] == "#":
        note = sharp_to_flat(note[:2]) + note[2:]

    sound_data = wavfile.read(os.path.join(notes_dir, f"{note}.wav"))
    return sound_data

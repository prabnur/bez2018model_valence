import os
from .bez2018model import nervegram, get_ERB_cf_list
from scipy.io import wavfile 
import numpy as np
from multiprocessing import Pool, cpu_count

NOTES_DIR = "../Iowa Notes/Mono"
SPIKES_DIR = "./spikes/mono"


def generate_spikes_sync(sound_data, duration=0.25):
    np.random.seed(711)

    # get data
    input_signal_fs, input_signal = sound_data
    input_signal = input_signal[:int(duration*input_signal_fs)]
    sponts = np.load("./model/sponts.npy")

    result = nervegram(
        input_signal,
        input_signal_fs,

        # Params worth changing
        max_spikes_per_train=100,
        nervegram_fs=20e3,

        # Params not worth changing
        synapseMode=1, # 0 = less computation
        species=2, # Human (Shera et al. 2002)
        num_cf=3500, min_cf=125, max_cf=16e3,
        spont_list=sponts,
        implnt=1, # 0 = approximate, 1 = actual Power Law
        num_spike_trains=1,

        # What is returned
        return_vihcs=False,
        return_meanrates=False,
        return_spike_times=True,
        return_spike_tensor_sparse=False,
        return_spike_tensor_dense=False,
    )
    return np.array(result["nervegram_spike_times"])

def save_spikes(note):
    sound_data = wavfile.read(os.path.join(NOTES_DIR, f"{note}.wav"))
    spike_times = generate_spikes_sync(sound_data)
    np.save(os.path.join(SPIKES_DIR, f"{note}.npy"), spike_times)

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

def get_spikes(note):
    if note[1] == "#":
        note = sharp_to_flat(note[:2]) + note[2:]
    return np.load(os.path.join(SPIKES_DIR, f"{note}.npy"))

def generate_scale(scale):
    processed = [
        filename.split(".")[0] 
            for filename in os.listdir(SPIKES_DIR)
                if filename.endswith(".npy")
    ]

    for filename in os.listdir(NOTES_DIR):
        if (filename.endswith(".wav") and (not filename.startswith("."))):
            note = filename.split(".")[0]
            if not (scale == int(note[-1])):
                print(f"Skipping: {note}")
                continue
            print(f"Processing: {note}")
            if note not in processed:
                save_spikes(note)
            else:
                print(f"Already processed: {note}")

    print(f"\nDONE WITH SCALE {scale}\n")


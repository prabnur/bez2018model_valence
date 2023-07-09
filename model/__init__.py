import os
from .bez2018model import nervegram
# from .num_spike_trains import NUM_SPIKE_TRAINS_LIST
from scipy.io import wavfile 
import numpy as np

NOTES_DIR = "../Iowa Notes/Mono"
SPIKES_DIR = "./spikes/mono"

def generate_spikes(sound_data, duration=0.25):
    np.random.seed(711)

    # get data
    input_signal_fs, input_signal = sound_data
    input_signal = input_signal[:int(duration*input_signal_fs)]

    result = nervegram(
        input_signal,
        input_signal_fs,

        # Params worth changing
        synapseMode=1, # 0 = less computation
        max_spikes_per_train=200,
        nervegram_fs=20e3,

        # Params not worth changing
        species=2, # Human (Shera et al. 2002)
        num_cf=3500,
        min_cf=125,
        max_cf=16e3,
        # num_spike_trains_list=NUM_SPIKE_TRAINS_LIST,
        num_spike_trains=15,
        implnt=1, # 0 = approximate, 1 = actual Power Law

        # What is returned
        return_vihcs=False,
        return_meanrates=False,
        return_spike_times=True,
        return_spike_tensor_sparse=False,
        return_spike_tensor_dense=False,
    )
    return result["nervegram_spike_times"]


def save_spikes(note):
    sound_data = wavfile.read(os.path.join(NOTES_DIR, f"{note}.wav"))
    spike_times = generate_spikes(sound_data)
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

def generate_all():
    processed = [
        filename.split(".")[0] 
            for filename in os.listdir(SPIKES_DIR)
                if filename.endswith(".npy")
    ]

    for filename in os.listdir(NOTES_DIR):
        if filename.endswith(".wav"):
            note = filename.split(".")[0]
            scale = int(note[-1])
            if not (scale == 3 or scale == 5):
                print(f"Skipping: {note}")
                continue
            print(f"Processing: {note}")
            if note not in processed:
                save_spikes(note)
            else:
                print(f"Already processed: {note}")

    print("\nDONE WITH SCALE 3 & 5\n")

    processed = [
        filename.split(".")[0] 
            for filename in os.listdir(SPIKES_DIR)
                if filename.endswith(".npy")
    ]

    for filename in os.listdir(NOTES_DIR):
        if filename.endswith(".wav"):
            note = filename.split(".")[0]
            print(f"Processing: {note}")
            if note not in processed:
                save_spikes(note)
            else:
                print(f"Already processed: {note}")

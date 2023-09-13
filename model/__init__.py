from fileinput import filename
from math import inf
import os
from shlex import join
from .bez2018model import nervegram, get_ERB_cf_list
from scipy.io import wavfile 
import numpy as np
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

NOTES_DIR = "../Iowa Notes/Mono"
SPIKES_DIR = "./spikes/mono"
SPIKES_ABS_DIR = "./spikes/mono/abs"
TONE_SPIKES_DIR = "./spikes/tone"

MAX_VAL = 2**15

def normalize_to_range(array):
    # Find the minimum and maximum values in the array
    min_val = np.min(array)
    max_val = np.max(array)
    
    # Handle the edge case where min_val == max_val
    if min_val == max_val:
        return np.zeros_like(array, dtype=float)
    
    # Normalize the array to the range [-1, 1]
    normalized_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return normalized_array

def normalize_absolute(array):
    # Find the minimum and maximum values in the array
    min_val = np.min(array)
    max_val = np.max(array)
    
    # Handle the edge case where min_val == max_val
    if min_val == max_val:
        return np.zeros_like(array, dtype=float)
    
    # Normalize the array to the range [-1, 1]
    normalized_array = array / MAX_VAL
    return normalized_array

def generate_spikes_sync(sound_data, duration=0.25):
    np.random.seed(711)

    # get data
    input_signal_fs, input_signal = sound_data
    input_signal = input_signal[:int(duration*input_signal_fs)]
    sponts = np.load("./model/sponts.npy")

    # input_signal = normalize_to_range(input_signal)
    input_signal = normalize_absolute(input_signal)

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
    spike_times = spike_times[0]
    if spike_times.shape != (3500,18,100):
        print(f"ERROR: {note} has shape {spike_times.shape}")

    path = os.path.join(SPIKES_ABS_DIR, f"{note}.npy")
    # Extract the directory name from the file path
    directory = os.path.dirname(path)
    
    # Create the directory recursively if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the numpy array to the path
    np.save(path, spike_times)

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

def get_spikes_abs(note):
    if note[1] == "#":
        note = sharp_to_flat(note[:2]) + note[2:]
    return np.load(os.path.join(SPIKES_ABS_DIR, f"{note}.npy"))

def generate_scale(scale):
    processed = [
        filename.split(".")[0] 
            for filename in os.listdir(SPIKES_ABS_DIR)
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

def generate_pure_tone(frequency, duration=0.25, sample_rate=44100):
    """
    Generate a pure tone of a given frequency and duration.

    Parameters:
    - frequency (float): The frequency of the tone in Hz.
    - duration (float): The duration of the tone in seconds.
    - sample_rate (int): The sample rate in Hz. Default is 44100.

    Returns:
    - numpy.ndarray: A NumPy array containing the samples of the pure tone.
    """
    # Create an array of time values
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate the samples using the sine function
    samples = np.sin(2 * np.pi * frequency * t)
    
    return samples


def save_spikes_tone(freq):
    tone = generate_pure_tone(freq)
    spike_times = generate_spikes_sync((44100, tone))
    if spike_times.shape == (1, 3500,18,100):
        spike_times = spike_times[0]
    if spike_times.shape != (3500,18,100):
        print(f"ERROR: {freq} tone spikes has shape {spike_times.shape}")
    np.save(os.path.join(TONE_SPIKES_DIR, f"{freq}.npy"), spike_times)

def get_tone_spikes(freq, attempts=0):
    max_attempts = 3  # Maximum number of attempts
    
    try:
        spikes = np.load(os.path.join(TONE_SPIKES_DIR, f"{freq}.npy"))
        return spikes
    
    except OSError:
        if attempts >= max_attempts:
            print(f"Failed to get spikes for frequency {freq} after {max_attempts} attempts.")
            return None
        
        save_spikes_tone(freq)
        return get_tone_spikes(freq, attempts + 1)


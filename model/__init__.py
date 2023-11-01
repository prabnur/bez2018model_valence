import os
from .bez2018model import nervegram
from scipy.io import wavfile
import numpy as np
from cache import NOTES_DIR, RNG_SPIKES_DIR, SPIKES_DIR, load_note_sound, save, sharp_to_flat

MAX_VAL = 2**15


def normalize_to_range(array):
    # Find the minimum and maximum values in the array
    min_val = np.min(array)
    max_val = np.max(array)

    # Handle the edge case where min_val == max_val
    if min_val == max_val:
        return np.zeros_like(array, dtype=float)

    # Normalize the array to the range [-1, 1]
    normalized_array = (2 * ((array - min_val) / (max_val - min_val))) - 1
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


def generate_spikes_sync(sound_data, duration=0.25, seed=711):
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
        synapseMode=1,  # 0 = less computation
        species=2,  # Human (Shera et al. 2002)
        num_cf=3500, min_cf=125, max_cf=16e3,
        spont_list=sponts,
        implnt=1,  # 0 = approximate, 1 = actual Power Law
        num_spike_trains=1,
        random_seed=seed,

        # What is returned
        return_vihcs=False,
        return_meanrates=False,
        return_spike_times=True,
        return_spike_tensor_sparse=False,
        return_spike_tensor_dense=False,
    )
    return np.array(result["nervegram_spike_times"])


def save_spikes(note, instrument="piano"):
    notes_dir = f"../notes/{instrument}"
    spikes_dir = f"./spikes/{instrument}"

    sound_data = wavfile.read(os.path.join(notes_dir, f"{note}.wav"))
    spike_times = generate_spikes_sync(sound_data)
    spike_times = spike_times[0]
    if spike_times.shape != (3500, 18, 100):
        print(f"ERROR: {note} has shape {spike_times.shape}")

    path = os.path.join(spikes_dir, f"{note}.npy")
    save(path, spike_times)


def first_n_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1
    return primes


def save_spikes_rng(note, seed):
    sound_data = load_note_sound(note)

    file_location = os.path.join(RNG_SPIKES_DIR, note, f"{note}_{seed}.npy")
    if os.path.exists(file_location):
        return False

    print(f"Processing {note} {seed}")

    spike_times = generate_spikes_sync(sound_data, seed=seed)
    spike_times = spike_times[0]
    if spike_times.shape != (3500, 18, 100):
        print(f"ERROR: {note} {seed} has shape {spike_times.shape}")

    save(file_location, spike_times)
    return True


def get_decoded_exp(note, normalize=True):
    if note[1] == "#":
        note = sharp_to_flat(note[:2]) + note[2:]
    decoded = np.load(os.path.join("./decoded/exp", f"{note}.npy"))
    if normalize:
        decoded = normalize_to_range(decoded)
    return decoded


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

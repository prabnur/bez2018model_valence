from model import generate_spikes_sync
import numpy as np
import os
from cache import SPIKES_DIR

TONE_SPIKES_DIR = os.path.join(SPIKES_DIR, "tone")

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

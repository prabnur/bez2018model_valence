import os
from scipy.io import wavfile
from cache import save

from model import generate_spikes_sync

if __name__ == "__main__":
    sound_data = wavfile.read("/Users/prabnurbal/Documents/notes/white_noise.wav")
    spike_times = generate_spikes_sync(sound_data)
    if spike_times.shape != (3500, 18, 100):
        spike_times = spike_times[0]
    if spike_times.shape != (3500, 18, 100):
        print(f"ERROR: shape {spike_times.shape}")
    save(os.path.join("./spikes", "white_noise.npy"), spike_times)

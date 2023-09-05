import matplotlib.pyplot as plt
from model.bez2018model import get_ERB_cf_list
import numpy as np
from scipy.io import wavfile

BASE = 32
cf_list = get_ERB_cf_list(num_cf=3500, min_cf=125, max_cf=16e3)

def decode(conc_prof):
    decoded = 0
    for i, count in enumerate(conc_prof):
        # if i <= 1:
        #     continue
        decoded += (BASE**(i+1))*count
    return decoded

def plot_concurrency(conc_profiles):
    decoded = [decode(conc_prof) for conc_prof in conc_profiles]
    # Plot line chart cf_list on X axis and decoded on Y axis

    plt.plot(cf_list, decoded)
    plt.xlabel("CF")
    plt.ylabel("Decoded")
    plt.xlim([0, 4000])
    plt.show()

def fourier_transform(note, shouldPlot):
    # Read the WAV file
    sample_rate, data = wavfile.read(f'/Users/prab/Documents/Play/Iowa Notes/Mono/{note}.wav')

    # If the audio file has multiple channels (e.g., stereo), take one channel
    # if len(data.shape) == 2:
    #     data = data[:, 0]

    # Perform the Fourier Transformß
    frequencies = np.fft.fftfreq(len(data), 1/sample_rate)
    positive_freq_idxs = np.where(frequencies > 0)
    frequencies = frequencies[positive_freq_idxs]
    fourier_transform = np.fft.fft(data)
    fourier_transform = fourier_transform[positive_freq_idxs]

    if shouldPlot:
        # Plot the Fourier Transform
        plt.figure(figsize=(10, 5))
        plt.title('Fourier Transform')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.plot(frequencies, np.abs(fourier_transform))
        plt.xlim([0, 4000])
        plt.show()

    return frequencies, np.abs(fourier_transform)

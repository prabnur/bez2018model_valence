from model import generate_scale, save_spikes

NOTES_DIR = "../Iowa Notes/Mono"

# Stereo
# (num_spike_trains, CFs, num_channels (2), spikes_per_train)
# eg. (40, 1, 2, 100)

# Mono
# (num_spike_trains, CFs, spikes_per_train)
# wg. (40, 1, 100)

if __name__ == "__main__":
    note = "C4"
    save_spikes(note)
    # generate_scale(4)
    # generate_scale(3)
    # generate_scale(5)

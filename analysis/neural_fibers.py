import numpy as np
import matplotlib.pyplot as plt

def analyze_neural_fibers(fiber_array, observation_time=0.25):
    """
    Analyze an array of neural fibers of an inner-hair cell and plot spike times.
    
    Parameters:
        fiber_array (list of np.array): A list of numpy arrays, each containing spike times for a single fiber.
        observation_time (float): The time window over which spikes are observed, in seconds.
        
    Returns:
        mean_spike_time (float): The mean spike time across all fibers.
        std_spike_time (float): The mean standard deviation of spike times across all fibers.
        num_spikes (float): The mean number of spikes across all fibers.
        spike_rate (float): The mean spike rate across all fibers.
    """
    
    # Initialize variables to store metrics
    mean_spike_times = []
    std_spike_times = []
    num_spikes_list = []
    
    # Initialize a figure for plotting
    plt.figure(figsize=(12, 8))
    
    # Loop through each fiber to collect metrics and plot spikes
    for i, spikes in enumerate(fiber_array):
        if len(spikes) > 0:  # Check if the array is non-empty
            mean_spike_times.append(np.mean(spikes))
            std_spike_times.append(np.std(spikes))
        else:
            mean_spike_times.append(0)
            std_spike_times.append(0)
        
        num_spikes_list.append(len(spikes))
        
        plt.scatter(spikes, [i]*len(spikes), marker='|')
        
    # Calculate final metrics
    mean_spike_time = np.nanmean(mean_spike_times)  # Using np.nanmean to handle NaNs
    std_spike_time = np.nanmean(std_spike_times)  # Using np.nanmean to handle NaNs
    num_spikes = np.mean(num_spikes_list)
    spike_rate = num_spikes / observation_time
    
    # Plotting
    plt.xlabel("Time (s)")
    plt.ylabel("Fiber index")
    plt.title("Spike Times Across Neural Fibers")
    plt.show()
    
    print(f"Mean Spike Time: {mean_spike_time:.3f} s")
    print(f"Standard Deviation of Spike Times: {std_spike_time:.3f} s")
    print(f"Mean Number of Spikes: {num_spikes:.1f}")
    print(f"Mean Spike Rate: {spike_rate:.1f} spikes/s")
    print(f"Observation Time: {observation_time} s")
    
    return mean_spike_time, std_spike_time, num_spikes, spike_rate
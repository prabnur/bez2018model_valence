from analysis.musical import calculate_intervals, CONSONANCE_ORDER
from model import get_spikes_abs
root_note = "C4"
intvls = calculate_intervals(root_note)

most_consonant = intvls[CONSONANCE_ORDER[0]]
least_consonant = intvls[CONSONANCE_ORDER[-1]]

root_spikes = get_spikes_abs(root_note)
most_consonant_spikes = get_spikes_abs(most_consonant)
least_consonant_spikes = get_spikes_abs(least_consonant)

from analysis.temporal import create_concurrency_profile

root_profile = [create_concurrency_profile(cf) for cf in root_spikes]
most_consonant_profile = [create_concurrency_profile(cf) for cf in most_consonant_spikes]
least_consonant_profile = [create_concurrency_profile(cf) for cf in least_consonant_spikes]

from analysis.spectral import decode
import numpy as np
BASE = 8

root_decoded = [decode(conc_prof, BASE) for conc_prof in root_profile]
most_consonant_decoded = [decode(conc_prof, BASE) for conc_prof in most_consonant_profile]
least_consonant_decoded = [decode(conc_prof, BASE) for conc_prof in least_consonant_profile]

# Plot root_decoded and most_consonant_decoded in seprate colours with most_consonant_decoded overlapping root_decoded and a wide figure size

x = np.arange(len(root_decoded))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Spacing between layers
z_offset = 5

# Plotting the second array with a Z-offset
ax.plot(x, most_consonant_decoded, zs=0, zdir='z', label='most_consonant_decoded', color='green')


# Plotting the first array
ax.plot(x, root_decoded, zs=z_offset, zdir='z', label='root_decoded', color='blue')


# Plotting the third array with twice the Z-offset
ax.plot(x, least_consonant_decoded, zs=2 * z_offset, zdir='z', label='least_consonant_decoded', color='red')

# Labels and title
ax.set_xlabel('Characteristic Frequency')
ax.set_ylabel('Decoded Value')
ax.set_zlabel('Layer')
ax.set_title('3D plot of Decoded Concurrency Profiles')

# Legend
ax.legend()

# Show the plot
plt.show()

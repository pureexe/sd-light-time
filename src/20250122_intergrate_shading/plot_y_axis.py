from sh_utils import get_uniform_rays_dense_top, get_uniform_rays


import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual arrays)
spread_rays = get_uniform_rays(1,1,10000)[0,0,:,2]
dense_rays = get_uniform_rays_dense_top(1,1,10000)[0,0,:,2]

# Create a histogram
plt.hist(spread_rays, bins=100, alpha=0.5, label='spread_rays')
plt.hist(dense_rays, bins=100, alpha=0.5, label='dense_rays')

# Add title and labels
plt.title('Histogram in z-axis of spread_rays and dense_rays')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show legend
plt.legend()

# Save the plot as PNG
plt.savefig('histogram.png', format='png')

# Optionally, display the plot
# plt.show()

# Close the plot to free memory
plt.close()
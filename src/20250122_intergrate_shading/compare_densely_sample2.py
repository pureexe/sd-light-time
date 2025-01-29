from sh_utils import get_uniform_rays_dense_top, get_uniform_rays


import matplotlib.pyplot as plt
import numpy as np
# Function to create heatmap from 3D data projections
def create_heatmap(data, ax, xlabel, ylabel, label, x_bins=100, y_bins=100):
    x = data[:, 0]
    y = data[:, 1]
    
    # Define the range for x and y
    x_range = [-1, 1]
    y_range = [-1, 1]
    
    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins], range=[x_range, y_range])
    
    # Plot the heatmap
    im = ax.imshow(hist.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{xlabel}-{ylabel} Projection Heatmap ({label})')
    plt.colorbar(im, ax=ax)

# Assuming dense_rays and spread_rays are defined by your functions
def main():
    # Example data, replace this with the actual data fetching functions
    dense_rays = get_uniform_rays_dense_top(1, 1, 1000000)[0, 0, :, :3]
    spread_rays = get_uniform_rays(1, 1, 1000000)[0, 0, :, :3]
    
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))  # 3 rows and 2 columns

    # Plot x-y projections for dense and spread
    create_heatmap(dense_rays[:, :2], axs[0, 0], 'x', 'y', 'Dense')
    create_heatmap(spread_rays[:, :2], axs[0, 1], 'x', 'y', 'Spread')
    
    # Plot x-z projections for dense and spread
    create_heatmap(dense_rays[:, [0, 2]], axs[1, 0], 'x', 'z', 'Dense')
    create_heatmap(spread_rays[:, [0, 2]], axs[1, 1], 'x', 'z', 'Spread')
    
    # Plot y-z projections for dense and spread
    create_heatmap(dense_rays[:, [1, 2]], axs[2, 0], 'y', 'z', 'Dense')
    create_heatmap(spread_rays[:, [1, 2]], axs[2, 1], 'y', 'z', 'Spread')

    plt.tight_layout()
    
    # Save the figure as a PNG file
    plt.savefig('ray_heatmaps_3x2_with_labels.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

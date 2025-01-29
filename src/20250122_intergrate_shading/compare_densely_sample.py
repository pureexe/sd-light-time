from sh_utils import get_uniform_rays_dense_top, get_uniform_rays
import matplotlib.pyplot as plt 

def plot_rays(dense_rays, spread_rays, filename="rays_comparison.png"):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    titles = ["X-Y Plane", "X-Z Plane", "Y-Z Plane"]
    labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    
    for i, (ax_dense, ax_spread) in enumerate(zip(axes[0], axes[1])):
        x_dense, y_dense = dense_rays[:, i], dense_rays[:, (i+1) % 3]
        x_spread, y_spread = spread_rays[:, i], spread_rays[:, (i+1) % 3]
        
        ax_dense.scatter(x_dense, y_dense, s=1, color='blue', alpha=0.5)
        ax_spread.scatter(x_spread, y_spread, s=1, color='red', alpha=0.5)
        
        ax_dense.set_xlim([-1, 1])
        ax_dense.set_ylim([-1, 1] if i < 2 else [0, 1])
        ax_spread.set_xlim([-1, 1])
        ax_spread.set_ylim([-1, 1] if i < 2 else [0, 1])
        
        ax_dense.set_title(f"Dense Rays - {titles[i]}")
        ax_spread.set_title(f"Spread Rays - {titles[i]}")
        
        ax_dense.set_xlabel(labels[i][0])
        ax_dense.set_ylabel(labels[i][1])
        ax_spread.set_xlabel(labels[i][0])
        ax_spread.set_ylabel(labels[i][1])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    dense_rays = get_uniform_rays_dense_top(1,1,5000)[0,0]
    spread_rays = get_uniform_rays(1,1,5000)[0,0]
    plot_rays(dense_rays, spread_rays)
    # plot compare to plt

if __name__ == "__main__":
    main()
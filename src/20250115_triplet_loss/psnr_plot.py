import os
import matplotlib.pyplot as plt

def plot_psnr_from_epochs(base_dir, output_image='psnr_plot.png'):
    """
    Reads PSNR values from text files in epoch directories, averages them, and plots the result.

    Parameters:
        base_dir (str): Base directory to search for "epoch_" directories.
        output_image (str): File name to save the resulting plot.
    """
    epoch_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("epoch_")
    ]

    # Parse epoch numbers and sort by epoch number
    epoch_data = []
    for dir_name in epoch_dirs:
        try:
            epoch_num = int(dir_name.split('_')[1])
            epoch_data.append((epoch_num, dir_name))
        except (IndexError, ValueError):
            print(f"Skipping invalid directory name: {dir_name}")

    epoch_data.sort(key=lambda x: x[0])

    # Collect average PSNR values
    epochs = []
    psnr_averages = []

    for epoch_num, dir_name in epoch_data:
        psnr_dir = os.path.join(base_dir, dir_name, "psnr")
        if not os.path.exists(psnr_dir):
            print(f"Skipping {dir_name}: PSNR directory not found.")
            continue

        psnr_values = []
        for file_name in os.listdir(psnr_dir):
            if file_name.startswith("everett") and file_name.endswith(".txt"):
                file_path = os.path.join(psnr_dir, file_name)
                try:
                    with open(file_path, 'r') as f:
                        value = float(f.read().strip())
                        psnr_values.append(value)
                except ValueError:
                    print(f"Skipping invalid PSNR value in file: {file_path}")

        if psnr_values:
            avg_psnr = sum(psnr_values) / len(psnr_values)
            epochs.append(epoch_num)
            psnr_averages.append(avg_psnr)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, psnr_averages, marker='o', linestyle='-', color='b')
    plt.title("Average PSNR by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average PSNR")
    plt.grid(True)
    plt.savefig(output_image)
    plt.close()
    print(f"Plot saved to {output_image}")

# Example usage
# plot_psnr_from_epochs("/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250104/multi_mlp_fit/lightning_logs/version_95350")

plot_psnr_from_epochs("/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250104/multi_mlp_fit/lightning_logs/version_95354")
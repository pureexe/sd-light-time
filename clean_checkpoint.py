import os
import re

def remove_checkpoints(root_dir):
    # Regex pattern to match files in format epoch={:06d}.ckpt
    pattern = re.compile(r'epoch=(\d{6})\.ckpt')

    for subdir, dirs, files in os.walk(root_dir):
        if "checkpoints" in subdir:
            epoch_files = []
            # Collect all epoch files
            for file in files:
                match = pattern.match(file)
                if match:
                    epoch_num = int(match.group(1))
                    epoch_files.append((epoch_num, file))

            # If no epoch files are found, skip the directory
            if not epoch_files:
                continue

            # Find the file with the largest number
            largest_file = max(epoch_files, key=lambda x: x[0])

            # Filter out files that are mod 20 == 19 or largest file
            keep_files = [file for epoch_num, file in epoch_files if epoch_num % 20 == 19 or file == largest_file[1]]

            # Remove all other files
            for epoch_num, file in epoch_files:
                if file not in keep_files:
                    file_path = os.path.join(subdir, file)
                    #print(f"Removing file: {file_path}")
                    os.remove(file_path)

# Example usage
root_directory = "output/20240918/multi_mlp_fit/lightning_logs"
remove_checkpoints(root_directory)
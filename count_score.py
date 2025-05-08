import os

def read_floats_from_file(filepath):
    with open(filepath, 'r') as file:
        return [float(line.strip()) for line in file if line.strip()]

def compute_average_from_directory(directory):
    all_floats = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            try:
                numbers = read_floats_from_file(filepath)
                all_floats.extend(numbers)
            except ValueError:
                print(f"Skipping file with non-numeric content: {filename}")
    
    if all_floats:
        average = sum(all_floats) / len(all_floats)
        print(f"Average: {average}")
    else:
        print("No valid floating point numbers found.")

# Example usage
directory_path = '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250419_train_in_the_wild/val_rotate_everett_kitchen6/run1/1.0/no_clip/1e-4/chk19/lightning_logs/version_0/psnr'  # Replace with your directory
compute_average_from_directory(directory_path)
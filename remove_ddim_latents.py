import os
import shutil

def remove_ddim_latents(base_dir):
    """
    Recursively search for and remove directories named 'ddim_latents'.

    :param base_dir: The base directory to start the search.
    """
    for root, dirs, files in os.walk(base_dir, topdown=False):  # Traverse bottom-up to safely remove directories
        for dir_name in dirs:
            if dir_name == "ddim_latents":
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
                except Exception as e:
                    print(f"Failed to remove {dir_path}: {e}")

# Usage example
base_directory = "output/20241027"  # Replace with your base directory path
remove_ddim_latents(base_directory)
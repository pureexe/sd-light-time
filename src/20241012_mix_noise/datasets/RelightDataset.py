from datasets.base_relight_dataset import BaseRelightDataset
import os 
import torch

ACCEPT_EXTENSION = ('jpg', 'png', 'jpeg', 'exr')

class RelightDataset(BaseRelightDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # create index_file
        self.index_file = self.get_avaliable_images()
        self.image_index = self.index_file
        self.files = self.image_index

    def get_avaliable_images(self, directory_path = None, accept_extensions=ACCEPT_EXTENSION):
        """
        Recursively get the list of files from a directory that match the accepted extensions.
        The file list is sorted by directory and then by filename.

        Args:
        - directory_path (str): Path to the directory to search.
        - accept_extensions (tuple or list): Tuple or list of acceptable file extensions (e.g., ('.jpg', '.png')).

        Returns:
        - List of sorted file paths that match the accepted extensions.
        """

        # Set default directory path
        if directory_path is None:
            directory_path = os.path.join(self.root_dir, "images")

        matched_files = []

        # Walk through directory and subdirectories
        for root, directory, files in os.walk(directory_path):
            # Filter files by extension
            for file in sorted(files):
                if file.lower().endswith(accept_extensions):
                    filepath = os.path.join(root, file)
                    relative_path = os.path.relpath(filepath, directory_path)
                    filename = os.path.splitext(relative_path)[0]
                    matched_files.append(filename)

        # Sort matched files by directory and filename
        matched_files =  sorted(matched_files)
        return matched_files
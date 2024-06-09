import torch
class FaceLeftRightDataset(torch.utils.data.Dataset):
    
    def __init__(self, num_files=1, root_dir=DATASET_ROOT_DIR, split="train", val_hold=100, val_fly=5, *args, **kwargs) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_files = num_files
        self.val_hold = val_hold
        self.val_fly = val_fly
        self.files, self.subdirs = self.+()

        if split == "train":
            self.files = self.files[val_hold:]
            self.subdirs = self.subdirs[val_hold:]
        elif split == "val":
            self.files = self.files[:val_fly] + self.files[val_hold:val_hold+val_fly] 
            self.subdirs = self.subdirs[:val_fly] + self.subdirs[val_hold:val_hold+val_fly]

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
            torchvision.transforms.Resize(512),  # Resize the image to 512x512
        ])
        # read prompt 
        with open(os.path.join(DATASET_ROOT_DIR, "prompts.json")) as f:
            self.prompt = json.load(f)

    def get_image_files(self):
        files = []
        subdirs = []
        for subdir in sorted(os.listdir(os.path.join(self.root_dir, "images"))):
            for fname in sorted(os.listdir(os.path.join(self.root_dir, "images", subdir))):
                if fname.endswith(".png"):
                    fname = fname.replace(".png","")
                    files.append(fname)
                    subdirs.append(subdir)
        return files, subdirs

    def split_dataset(self, split):
        
    def __len__(self):
        return len(self.files)
    
    def convert_to_grayscale(self, v):
        """convert RGB to grayscale

        Args:
            v (np.array): RGB in shape of [3,...]
        Returns:
            np.array: gray scale array in shape [...] (1 dimension less)
        """
        assert v.shape[0] == 3
        return 0.299*v[0] + 0.587*v[1] + 0.114*v[2]

    def get_light_direction(self, idx):
        light = np.load(os.path.join(self.root_dir, "light", self.subdirs[idx], f"{self.files[idx]}_light.npy")) 
        light = self.convert_to_grayscale(light.transpose())
        assert len(light) == 9
        if light[1] < 0.0:
            return 0 #left
        else:
            return 1 #right
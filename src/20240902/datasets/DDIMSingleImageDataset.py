from datasets.DDIMDataset import DDIMDataset
import torchvision 
import os
import torch

class DDIMSingleImageDataset(DDIMDataset):

    def __init__(self, image_path, control_paths, source_env_ldr, source_env_under, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 
        self.source_image_path = image_path
        self.rgb_image = self.transform['image'](self.get_image_from_path(image_path))
        self.source_env_ldr = self.transform['envmap'](self.get_image_from_path(source_env_ldr))
        self.source_env_under = self.transform['envmap'](self.get_image_from_path(source_env_under))
        # control_paths is a dictionary with key as the name of the control image and value as the path to the control image
        self.control_images = {key: self.transform['control'](self.get_image_from_path(value)) for key, value in control_paths.items()}

    def get_image_from_path(self, image_path):
        image = torchvision.io.read_image(image_path) / 255.0
        image = image[:3]
        # if image is one channel, repeat it to 3 channels
        if image.shape[0] == 1:
            image = torch.cat([image, image, image], dim=0)
        return image
       
    def get_item(self, idx, batch_idx):
        output = super().get_item(idx, batch_idx)
        output['source_image'] = self.rgb_image
        for key in self.control_images:
            output[key] = self.control_images[key]
        output['source_ldr_envmap'] = self.source_env_ldr
        output['source_norm_envmap'] = self.source_env_under
        output['name'] = os.path.basename(self.source_image_path).split(".")[0]
        return output
    
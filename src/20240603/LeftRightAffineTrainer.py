import os 
import torch 
import torchvision
import lightning as L
import numpy as np
import torchvision
import json
from constants import *
from UNet2DLeftRightAffineConditionModel import UNet2DLeftRightAffineConditionModel

from diffusers import StableDiffusionPipeline

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
args = parser.parse_args()
from tqdm.auto import tqdm
 

class LeftRightAffineDataset(torch.utils.data.Dataset):
    
    def __init__(self, num_files=1, root_dir=DATASET_ROOT_DIR, split="train", *args, **kwargs) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_files = num_files
        self.files, self.subdirs = self.get_image_files()

        if split == "train":
            self.files = self.files[100:]
            self.subdirs = self.subdirs[100:]
        elif split == "val":
            self.files = self.files[:10] + self.files[100:110] 
            self.subdirs = self.subdirs[:10] + self.subdirs[100:110]

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
        for subdir in sorted(os.listdir(os.path.join(DATASET_ROOT_DIR, "images"))):
            for fname in sorted(os.listdir(os.path.join(DATASET_ROOT_DIR, "images", subdir))):
                if fname.endswith(".png"):
                    fname = fname.replace(".png","")
                    files.append(fname)
                    subdirs.append(subdir)
        return files, subdirs

    def __len__(self):
        return len(self.files)
    
    def convert_to_grayscale(self, v):
        """convert RGB to grayscale

        Args:
            v (np.array): RGB in shape of [3,...]
        Returns:
            np.array: gray scale array in shape [...] (1 dimension less)
        """
        return 0.299*v[0] + 0.587*v[1] + 0.114*v[2]

    def get_light_direction(self, idx):
        light = np.load(os.path.join(self.root_dir, "light", self.subdirs[idx], f"{self.files[idx]}_light.npy")) 
        light = self.convert_to_grayscale(light.transpose())
        if light[1] < 0.0:
            return 0 #left
        else:
            return 1 #right
        
    def get_image(self, idx):
        image_path = os.path.join(self.root_dir, "images",  self.subdirs[idx], f"{self.files[idx]}.png")
        image = torchvision.io.read_image(image_path) / 255.0
        return image

    def __getitem__(self, idx):
        
        return {
            'name': self.files[idx],
            'pixel_values': self.transform(self.get_image(idx)),
            'light': self.get_light_direction(idx),
            'text': self.prompt[self.files[idx]]
        }

class LeftRightAffine(L.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        MASTER_TYPE = torch.float32
        # setup dataset
        affines = [
            # down-1
            torch.nn.Parameter(torch.ones(2, 320, 64, 64, dtype=MASTER_TYPE)),
            torch.nn.Parameter(torch.zeros(2, 320, 64, 64, dtype=MASTER_TYPE)),
            # down-2
            torch.nn.Parameter(torch.ones(2, 640, 32, 32, dtype=MASTER_TYPE)),
            torch.nn.Parameter(torch.zeros(2, 640, 32, 32, dtype=MASTER_TYPE)),
            # down-3
            torch.nn.Parameter(torch.ones(2, 1280, 16, 16, dtype=MASTER_TYPE)),
            torch.nn.Parameter(torch.zeros(2, 1280, 16, 16, dtype=MASTER_TYPE)),
            # up-3
            torch.nn.Parameter(torch.ones(2, 1280, 8, 8, dtype=MASTER_TYPE)),
            torch.nn.Parameter(torch.zeros(2, 1280, 8,8, dtype=MASTER_TYPE)),
            # up-2
            torch.nn.Parameter(torch.ones(2, 1280, 16, 16, dtype=MASTER_TYPE)),
            torch.nn.Parameter(torch.zeros(2, 1280, 16, 16, dtype=MASTER_TYPE)),
            # up-1
            torch.nn.Parameter(torch.ones(2, 640, 32, 32, dtype=MASTER_TYPE)),
            torch.nn.Parameter(torch.zeros(2, 640, 32, 32, dtype=MASTER_TYPE))
        ]
        # add small noise to parameter
        for p in affines:
            p.data += torch.randn_like(p, dtype=MASTER_TYPE) * 1e-3


        # load pipeline
        sd_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_path, torch_dtype=MASTER_TYPE)
        self.pipe.safty_checker = None
        # load unet from pretrain 
        self.pipe.unet = UNet2DLeftRightAffineConditionModel.from_pretrained(sd_path, subfolder='unet')
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        
        self.pipe.unet.set_affine(affines)

        
        self.affines_params = torch.nn.ParameterList(self.pipe.unet.get_affine_params())

        self.pipe.to('cuda')


    def training_step(self, batch, batch_idx):

        text_inputs = self.pipe.tokenizer(
                batch['text'],
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids


        latents = self.pipe.vae.encode(batch['pixel_values']).latent_dist.sample().detach()

        latents = latents * self.pipe.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long().to(latents.device)

        text_input_ids = text_input_ids.to(latents.device)
        encoder_hidden_states = self.pipe.text_encoder(text_input_ids)[0]

        # 
        target = noise 
    
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # set light direction
        assert batch['light'][0] in [0, 1]  # currently support only left and right
        assert len(batch['light']) == 1 #current support only batch size = 1
        self.pipe.unet.set_direction(batch['light'][0]) 

        model_pred = self.pipe.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.global_step and self.current_epoch % 20 == 0 and self.global_step > 9000 :
            prompts = []
            with open(os.path.join(DATASET_ROOT_DIR, "prompts.json")) as f:
                prompts = json.load(f)
            # generate 100 face image 
            for direction in [0,1]:
                print("GENERATING FACE DIRECTION: ", direction)
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/{direction}"
                os.makedirs(output_dir, exist_ok=True)
                self.pipe.unet.set_direction(direction)
                for seed in range(100):
                    image, _ = self.pipe(
                        prompts[f"{seed:05d}"], 
                        output_type="pil",
                        guidance_scale=7.5,
                        num_inference_steps=50,
                        return_dict = False,
                        generator=torch.Generator().manual_seed(seed)
                    )
                    image[0].save(f"{output_dir}/{seed:03d}.png")


        assert batch['light'][0] in [0, 1]  # currently support only left and right
        assert len(batch['light']) == 1 #current support only batch size = 1
        self.pipe.unet.set_direction(batch['light'][0]) 
        pt_image, _ = self.pipe(
            batch['text'], 
            output_type="pt",
            guidance_scale=7.5,
            num_inference_steps=50,
            return_dict = False,
            generator=torch.Generator().manual_seed(42)
        )
        gt_image = (batch["pixel_values"] + 1.0) / 2.0
        images = torch.cat([gt_image, pt_image], dim=0)
        image = torchvision.utils.make_grid(images, nrow=2, normalize=True, range=(0, 1))
        self.logger.experiment.add_image(f'image/{batch["name"][0]}', image, self.global_step)
        if self.global_step == 0 and batch_idx == 0:
            self.logger.experiment.add_text('text', batch['text'][0], self.global_step)
            self.logger.experiment.add_text('params', str(args), self.global_step)
            self.logger.experiment.add_text('learning_rate', str(args.learning_rate), self.global_step)
        return torch.zeros(1, )
    


    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        lora_layers = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        optimizer = torch.optim.Adam(lora_layers, lr=args.learning_rate)
        return optimizer



def main():
    model = LeftRightAffine()
    train_dataset = LeftRightAffineDataset(split="train")
    val_dataset = LeftRightAffineDataset(split="val")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    trainer = L.Trainer(max_epochs =1000, precision=32)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()
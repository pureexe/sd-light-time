import os 
import torch 
import torchvision
import lightning as L

from constants import *
from UNet2DSingleAffineConditionModel import UNet2DSingleAffineConditionModel

from diffusers import StableDiffusionPipeline

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
args = parser.parse_args()

 

class SingleAffineDataset(torch.utils.data.Dataset):
    def __init__(self, num_files=1, *args, **kwargs) -> None:
        super().__init__()
        self.num_files = num_files
        self.image = torchvision.io.read_image(f'{SRC}/man.png') / 255.0
        self.transform = torchvision.transforms.Compose([
            #torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
            torchvision.transforms.Resize(512),  # Resize the image to 512x512
        ])

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        return {
            'pixel_values': self.transform(self.image),
            'text': 'man wearing a gray sweater'
        }

class SingleAffine(L.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        # setup dataset
        self.affines = torch.nn.ParameterList([
            # down-1
            torch.nn.Parameter(torch.ones(1, 320, 64, 64)),
            torch.nn.Parameter(torch.zeros(1, 320, 64, 64)),
            # down-2
            torch.nn.Parameter(torch.ones(1, 640, 32, 32)),
            torch.nn.Parameter(torch.zeros(1, 640, 32, 32)),
            # down-3
            torch.nn.Parameter(torch.ones(1, 1280, 16, 16)),
            torch.nn.Parameter(torch.zeros(1, 1280, 16, 16)),
            # up-3
            torch.nn.Parameter(torch.ones(1, 1280, 8, 8)),
            torch.nn.Parameter(torch.zeros(1, 1280, 8,8)),
            # up-2
            torch.nn.Parameter(torch.ones(1, 1280, 16, 16)),
            torch.nn.Parameter(torch.zeros(1, 1280, 16, 16)),
            # up-1
            torch.nn.Parameter(torch.ones(1, 640, 32, 32)),
            torch.nn.Parameter(torch.zeros(1, 640, 32, 32))
        ])
        # add small noise to parameter
        for p in self.affines:
            p.data += torch.randn_like(p) * 1e-4


        # load pipeline
        sd_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_path, torch_dtype=torch.float32)
        self.pipe.safty_checker = None
        # load unet from pretrain 
        self.pipe.unet = UNet2DSingleAffineConditionModel.from_pretrained(sd_path, subfolder='unet')
        self.pipe.unet.set_affine(self.affines)
        self.pipe.to('cuda')


    def training_step(self, batch, batch_idx):


        self.pipe.unet.set_affine(self.affines)
        
        #input_ids = self.pipe.tokenizer(batch['text'], max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        text_inputs = self.pipe.tokenizer(
                batch['text'],
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids


        latents = self.pipe.vae.encode(batch['pixel_values']).latent_dist.sample()

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
        
        model_pred = self.pipe.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # get image from self.pipe and write to tensorboard
        self.pipe.unet.set_affine(self.affines)
        pt_image, _ = self.pipe(
            batch['text'], 
            output_type="pt",
            guidance_scale=7.5,
            num_inference_steps=50,
            return_dict = False,
            generator=torch.Generator().manual_seed(42)
        )
        self.logger.experiment.add_image('image', pt_image[0], self.global_step)
        for i, affine in enumerate(self.affines):
            self.log(f'sum_affine/{i}', torch.sum(affine[0]))
        if self.global_step == 0:
            self.logger.experiment.add_text('text', batch['text'][0], self.global_step)
            self.logger.experiment.add_text('params', str(args), self.global_step)
            self.logger.experiment.add_text('learning_rate', str(args.learning_rate), self.global_step)
        return torch.zeros(1, )
    


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        return optimizer



def main():
    model = SingleAffine()
    train_dataset = SingleAffineDataset(num_files=100)
    val_dataset = SingleAffineDataset(num_files=1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    trainer = L.Trainer(max_epochs =1000, precision=32)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()
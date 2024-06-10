import os 
import torch 
import torchvision
import lightning as L
import numpy as np
import torchvision
import json
from tqdm.auto import tqdm

from constants import *
from diffusers import StableDiffusionPipeline
from LightEmbedingBlock import set_light_direction, add_light_block
from FaceLeftRightDataset import FaceLeftRightDataset


import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
args = parser.parse_args()
 
 
class FaceLeftRightAffine(L.LightningModule):
    def __init__(self, learning_rate=1e-3, face100_every=10, *args, **kwargs) -> None:
        super().__init__()
        self.face100_every = face100_every
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        MASTER_TYPE = torch.float32


        # load pipeline
        sd_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_path, safety_checker=None, torch_dtype=MASTER_TYPE)
        #self.pipe.safety_checker = None
        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        
        add_light_block(self.pipe.unet)        
        self.pipe.to('cuda')
        # filter trainable layer
        unet_trainable = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.unet_trainable = torch.nn.ParameterList(unet_trainable)


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
        #self.pipe.unet.set_direction(batch['light'][0]) 
        set_light_direction(self.pipe.unet, batch['light'][0])

        model_pred = self.pipe.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

        self.log('train_loss', loss)

        return loss
    
    def generate_face100(self):
        prompts = []
        with open(os.path.join(DATASET_ROOT_DIR, "prompts.json")) as f:
            prompts = json.load(f)
        # generate 100 face image 
        for direction in [0,1]:
            print("GENERATING FACE DIRECTION: ", direction)
            output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/{direction}"
            os.makedirs(output_dir, exist_ok=True)
            set_light_direction(self.pipe.unet, direction)
            #self.pipe.unet.set_direction(direction)
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
                
    def generate_tensorboard(self, batch, batch_idx):
        set_light_direction(self.pipe.unet, batch['light'][0])
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
        image = torchvision.utils.make_grid(images, nrow=2, normalize=True)
        self.logger.experiment.add_image(f'image/{batch["name"][0]}', image, self.global_step)
        if self.global_step == 0 and batch_idx == 0:
            self.logger.experiment.add_text('text', batch['text'][0], self.global_step)
            self.logger.experiment.add_text('params', str(args), self.global_step)
            self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)
        
    def generate_tensorboard_guidance(self, batch, batch_idx):
        set_light_direction(self.pipe.unet, batch['light'][0])
        for guidance_scale in np.arange(1,11,0.5):
            pt_image, _ = self.pipe(
                batch['text'], 
                output_type="pt",
                guidance_scale=guidance_scale,
                num_inference_steps=50,
                return_dict = False,
                generator=torch.Generator().manual_seed(42)
            )
            gt_image = (batch["pixel_values"] + 1.0) / 2.0
            images = torch.cat([gt_image, pt_image], dim=0)
            image = torchvision.utils.make_grid(images, nrow=2, normalize=True)
            self.logger.experiment.add_image(f'guidance_scale/{guidance_scale:0.2f}', image, self.global_step)
        
    def test_step(self, batch, batch_idx):
        self.generate_face100()
        #self.generate_tensorboard(batch, batch_idx)
        self.generate_tensorboard_guidance(batch, batch_idx)


    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.current_epoch % self.face100_every == 0 and self.global_step > 1 :
            self.generate_face100()

        assert batch['light'][0] in [0, 1]  # currently support only left and right
        assert len(batch['light']) == 1 #current support only batch size = 1
        
        self.generate_tensorboard(batch, batch_idx)
        return torch.zeros(1, )
    


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        #optimizer = torch.optim.Adam(lora_layers, lr=args.learning_rate)
        return optimizer



def main():
    model = FaceLeftRightAffine(learning_rate=args.learning_rate)
    train_dataset = FaceLeftRightDataset(split="train")
    val_dataset = FaceLeftRightDataset(split="val")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    trainer = L.Trainer(max_epochs =1000, precision=32, check_val_every_n_epoch=1, default_root_dir=OUTPUT_MULTI)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()
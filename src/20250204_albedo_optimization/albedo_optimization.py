import torch
import numpy as np
import lightning as L
import diffusers
import torch
import os 
from PIL import Image 
import torchvision
import argparse 

MASTER_TYPE = torch.float32

parser = argparse.ArgumentParser()
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
parser.add_argument('-std_mul', '--std_multiplier', type=float, default=1e-4)
parser.add_argument('-lra', '--lr_albedo', type=float, default=1e-3)
parser.add_argument('-lrs', '--lr_shcoeff', type=float, default=1e-3)
parser.add_argument('--dataset_multipiler', type=int, default=1000)
parser.add_argument('--sh_regularize', type=float, default=1e-2)
args = parser.parse_args()


class MultiIluminationSceneDataset(torch.utils.data.Dataset):
    def __init__(self, scene_path = "path/to/scene", data_multiplier = 1, image_size = (512,512)):
        self.image_size = image_size
        self.load_images(scene_path)
        self.data_multiplier = data_multiplier # prevent dataloader reload too frequently.
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.image_size),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
        ])
        

    def load_images(self, scene_path):
        ACCEPT_EXTENSION = ['jpg', 'png']
        all_files =  [f for f in sorted(os.listdir(scene_path)) if f.split(".")[-1] in ACCEPT_EXTENSION]
        self.names = [".".join(f.split(".")[:-1]) for f in all_files]

        # read image 
        self.image_paths = [os.path.join(scene_path, f) for f in all_files]

    def get_num_images(self):
        return len(self.image_paths)

    def __len__(self):
        return self.data_multiplier * self.get_num_images()

    def __getitem__(self, idx):
        num_image = self.get_num_images()
        # we read image on the fly
        img = Image.open(self.image_paths[idx % num_image]).convert("RGB")  # Ensure 3 channels
        img = self.transform(img) # [3, H ,W]
        return {
            "id": idx % num_image,
            "name": self.names[idx % num_image],
            "image": img 
        }

class AlbedoOptimization(L.LightningModule):
    def __init__(self, num_images=25, image_size = (512,512), lr_albedo=1e-4, lr_shcoeff=1e-4):
        super().__init__()
        self.lr_shcoeff = lr_shcoeff
        self.lr_albedo = lr_albedo

        self.image_size = image_size
        self.num_images = num_images
        self.setup_albedo()
        self.setup_normal_pipeline()
        self.setup_spherical_coefficient()
        self.save_hyperparameters()

    def initial_with_mean(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.get_num_images(), shuffle=False, num_workers=8)
        for batch in dataloader:
            image = torch.mean(batch['image'],axis=0)
            self.albedo = torch.nn.Parameter(image[None])
            break

        
    def setup_normal_pipeline(self):
        self.pipe_normal = diffusers.MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
        ).to("cuda")


    def setup_albedo(self):
        """
        create the albedo tensor that can optimize shape [1,3,H,W]
        """
        self.albedo = torch.nn.Parameter(
            torch.randn(1,3, self.image_size[0], self.image_size[1]) * args.std_multiplier
        )
        #self.register_parameter("albedo", torch.nn.Parameter(self.albedo))

    
    def setup_spherical_coefficient(self):
        """
        create the spherical coefficient tensor that can optimize shape [num_images, 3, 9]
        """
        self.shcoeffs = torch.nn.Parameter(
            torch.randn(self.num_images, 3, 9) * args.std_multiplier
        )

        #self.register_parameter("shcoeffs", torch.nn.Parameter(self.shcoeffs))

        # setup constant factor 
        self.sh_constant = torch.tensor([
            1/np.sqrt(4*np.pi), 
            ((2*np.pi)/3)*(np.sqrt(3/(4*np.pi))), 
            ((2*np.pi)/3)*(np.sqrt(3/(4*np.pi))),
            ((2*np.pi)/3)*(np.sqrt(3/(4*np.pi))), 
            (np.pi/4)*(3)*(np.sqrt(5/(12*np.pi))), 
            (np.pi/4)*(3)*(np.sqrt(5/(12*np.pi))),
            (np.pi/4)*(3)*(np.sqrt(5/(12*np.pi))), 
            (np.pi/4)*(3/2)*(np.sqrt(5/(12*np.pi))), 
            (np.pi/4)*(1/2)*(np.sqrt(5/(4*np.pi)))]
        ).float()
        self.normal_map = None

    def compute_normal(self, image):
        # where X axis points right, Y axis points up, and Z axis points at the viewer
        # @see https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage
        normals = self.pipe_normal(
            image,
            output_type='pt'
        )
        return normals

    def get_normal(self, image):
        if self.normal_map is not None:
            return self.normal_map 
        assert len(image.shape) == 3 and image.shape[0] == 3 # make sure it feed only 1 image
        image = (image + 1.0) / 2.0
        normals = self.compute_normal(image)
        self.normal_map = normals.prediction[0]
        assert len(self.normal_map.shape) == 3 and self.normal_map.shape[0] == 3 #make sure self.normal_map is 3,H,W
        return self.normal_map

    def get_basis(self, normal):
        """
        get the basis function for the spherical harmonics
        @see https://github.com/diffusion-face-relighting/difareli_code/blob/2dd24a024f26d659767df6ecc8da4ba47c55e7a8/guided_diffusion/models/renderer.py#L25
        """
        # verify that we have normal shape [B,3,H,W]
        assert len(normal.shape) == 4 and normal.shape[1] == 3
        
        basis = torch.stack([
            normal[:,0]*0.+1.,                  # 1
            normal[:,0],                        # X
            normal[:,1],                        # Y
            normal[:,2],                        # Z
            normal[:,0] * normal[:,1],      # X*Y
            normal[:,0] * normal[:,2],          # X*Z
            normal[:,1] * normal[:,2],          # Y*Z
            normal[:,0]**2 - normal[:,1]**2,    # X**2 - Y**2
            3*(normal[:,2]**2) - 1,             # 3(Z**2) - 1
            ], 
            axis=1
        ) # [bz, 9, h, w]

        sh_constant = self.sh_constant[None, :, None, None].to(normal.device)
        
        basis = basis * sh_constant # [bz, 9, h, w]

        # verify that we use order 2 which has 9 basis 
        assert basis.shape[1] == 9
        return basis

    def render_image(self, shcoeffs, normal, albedo = None):
        """
        render image from normal using spherical harmonics and albedo
        O = albedo * \sum_{l,m} shcoeffs * BASIS(l,m,normal)
        """
        basis = self.get_basis(normal) # [bz, 9, h, w]
        shading = torch.sum(
            shcoeffs[:, :, :, None, None] # [bz, 3, 9, 1, 1]
            * basis[:, None, :, :, :], # [bz, None, 9, h, w]
            axis=2
        ) # [bz, 3, h, w]
        if albedo is not None:
            rendered = albedo * shading
        else:
            rendered = shading

        assert rendered.shape[1:] == normal.shape[1:] and shcoeffs.shape[0] == rendered.shape[0] # [bz, 3, h, w]
        return rendered

    def prepare_normal_map(self, batch):
        if self.normal_map is not None:
            return True

    def training_step(self, batch, batch_idx):
        bz = batch['image'].shape[0]
        assert bz == self.num_images # expect the batch size to be always same with num image to make thing easier 

        normal_map = self.get_normal(batch['image'][0])[None] #[1,3,H,W]

        render_image = self.render_image(self.shcoeffs, normal_map, self.albedo)

        loss = torch.nn.functional.mse_loss(render_image, batch['image'])

        loss += args.sh_regularize * torch.norm(self.shcoeffs, p=2) #add L2 regularize for stability

        self.log('train/loss', loss)

        return loss

    def log_tensorboard(self, batch, batch_idx):
        # save normal ball
        bz = batch['image'].shape[0]
        normal_ball_front, mask = get_ideal_normal_ball_y_up(256)
        normal_ball_front =  torch.tensor(normal_ball_front).to(batch['image'].device).permute(2,0,1)[None]
        self.logger.experiment.add_image(
            'ball_front/normal',
             n2v(normal_ball_front)[0],
            self.global_step
        )
        normal_ball_back = normal_ball_front.clone()
        normal_ball_back[:,2] *= -1 
        self.logger.experiment.add_image(
            'ball_back/normal',
            n2v(normal_ball_back)[0],
            self.global_step
        )
        # save normal map 
        normal_map = self.get_normal(batch['image'][0])
        self.logger.experiment.add_image(
            f'normal',
            n2v(normal_map),
            self.global_step
        )
        # save albedo 
        self.logger.experiment.add_image(
            f'albedo',
            n2v(self.albedo[0]),
            self.global_step
        )
        # save frontside chromeball to visualize the lighting 
        ball_image_front = self.render_image(self.shcoeffs, normal_ball_front)
        for i in range(bz):
            self.logger.experiment.add_image(
                f"ball_front/{batch['name'][i]}",
                n2v(ball_image_front[i]),
                self.global_step
            )
        # save backside chromeball to visualize the lighting 
        ball_image_back = self.render_image(self.shcoeffs, normal_ball_back)
        for i in range(bz):
            self.logger.experiment.add_image(
                f"ball_back/{batch['name'][i]}",
                n2v(ball_image_back[i]),
                self.global_step
            )
        # render shading
        render_shading = self.render_image(self.shcoeffs, normal_map[None])
        for i in range(bz):
            self.logger.experiment.add_image(
                f"shading/{batch['name'][i]}",
                n2v(render_shading[i]),
                self.global_step
            )
        # render image
        render_image = self.render_image(self.shcoeffs, normal_map[None], self.albedo)
        for i in range(bz):
            cat_image = torch.concatenate([batch['image'][i:i+1],render_image[i:i+1]],axis=0)
            cat_image = n2v(cat_image)
            self.logger.experiment.add_image(
                f"rendered/{batch['name'][i]}",
                torchvision.utils.make_grid(cat_image),
                self.global_step
            )
        mse = torch.nn.functional.mse_loss(render_image, batch['image'])
        psnr_value = 10 * torch.log10(1.0**2 / mse)
        self.log('val/render_psnr', psnr_value)

    def validation_step(self, batch, batch_idx):
        self.log_tensorboard(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.shcoeffs, 'lr': self.lr_shcoeff},
            {'params': self.albedo, 'lr': self.lr_albedo},
        ])
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

def get_ideal_normal_ball_y_up(size):
    # where X axis points right, Y axis points up, and Z axis points at the viewer
    x = torch.linspace(-1,1, size)
    y = torch.linspace(1,-1, size)
    y,x = np.meshgrid(y,x)

    # avoid negative value
    z2 = 1- x**2 - y ** 2
    mask = z2 >= 0
    
    # get real z value
    z = np.sqrt(np.clip(z2,0,1))

    x = x * mask
    y = y * mask
    z = z * mask

    # set z outside mask to be 1
    z = z + (1-mask)
    
    normal_map = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

    return normal_map, mask

def n2v(tensor):
    # normalize (-1,1) to visualize [0,1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor



def main():

    SCENE_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom10"
    train_dataset = MultiIluminationSceneDataset(
        scene_path=SCENE_DIR,
        data_multiplier=1000,
        image_size=(512,512)
    )
    val_dataset = MultiIluminationSceneDataset(
        scene_path=SCENE_DIR,
        data_multiplier=1,
        image_size=(512,512)
    )
    model = AlbedoOptimization(
        num_images = train_dataset.get_num_images(),
        lr_albedo = args.lr_albedo,
        lr_shcoeff = args.lr_shcoeff
    )
    model.initial_with_mean(val_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset.get_num_images(), shuffle=False, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset.get_num_images(), shuffle=False)
    trainer = L.Trainer(reload_dataloaders_every_n_epochs=0)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.checkpoint
    )


if __name__ == "__main__":
    main()


import torch
import numpy as np
import lightning as L
import diffusers
import torch
import os 
from PIL import Image 
import torchvision
import argparse 
import skimage 
from tonemapper import TonemapHDR
from natsort import natsorted

MASTER_TYPE = torch.float32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
    parser.add_argument('-std_mul', '--std_multiplier', type=float, default=1e-4)
    parser.add_argument('-lra', '--lr_albedo', type=float, default=1e-3)
    parser.add_argument('-lrs', '--lr_shcoeff', type=float, default=1e-3)
    parser.add_argument('--dataset_multipiler', type=int, default=1000)
    parser.add_argument('--sh_regularize', type=float, default=1e-3)
    parser.add_argument('--sh_3channel', type=float, default=0)
    parser.add_argument('--cold_start_albedo', type=int, default=0, help="epoch to start training albedo, 0 mean start training since first epoch")
    parser.add_argument('--use_lab', type=int, default=0)
    args = parser.parse_args()


class MultiIluminationSceneDataset(torch.utils.data.Dataset):
    def __init__(self, scene_path = "path/to/scene", data_multiplier = 1, image_size = (512,512), use_lab = False):
        self.use_lab = use_lab
        self.image_size = image_size
        self.load_images(scene_path)
        self.data_multiplier = data_multiplier # prevent dataloader reload too frequently.
        if self.use_lab:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.image_size),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.image_size),
            ])
        

    def load_images(self, scene_path):
        ACCEPT_EXTENSION = ['jpg', 'png']
        all_files =  [f for f in natsorted(os.listdir(scene_path)) if f.split(".")[-1] in ACCEPT_EXTENSION]
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

        if self.use_lab:
            img = np.array(img) #PIL to numpy range [0,255]
            img = img / 255.0 # rank [0,1]
            img = rgb2lab(img) # lab color sapce (0,1)

        img = self.transform(img) # [3, H ,W]

        return {
            "id": idx % num_image,
            "name": self.names[idx % num_image],
            "image": img.float()
        }

class AlbedoOptimization(L.LightningModule):
    def __init__(self, num_images=25, image_size = (512,512), lr_albedo=1e-4, lr_shcoeff=1e-4, std_multiplier = 1e-4, cold_start_albedo=0, sh_regularize=1e-3, sh_3channel=0, use_lab = False, log_shading=True, optimize_albedo=True):
        super().__init__()        
        self.sh_3channel = sh_3channel
        self.lr_shcoeff = lr_shcoeff
        self.lr_albedo = lr_albedo
        self.use_lab = use_lab
        self.sh_regularize = sh_regularize
        self.cold_start_albedo = cold_start_albedo
        self.std_multiplier = std_multiplier
        self.log_dir = ""
        self.log_shading = log_shading
        self.optimize_albedo = optimize_albedo
        
        self.image_size = image_size
        self.num_images = num_images
        self.setup_albedo()
        self.setup_normal_pipeline()
        self.setup_spherical_coefficient()
        self.save_hyperparameters()

        self.hdr2ldr = TonemapHDR(gamma=1.0, percentile=50, max_mapping=0.5)


    def initial_with_mean(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.get_num_images(), shuffle=False, num_workers=8)
        for batch in dataloader:
            if self.use_lab:
                n_images = []
                for a in batch['image']:
                    device = a.device
                    a = a.permute(1,2,0).cpu().numpy()
                    a = lab2rgb(a) # rgb color space [0,1]
                    a = torch.tensor(a).to(device)
                    a = a.permute(2,0,1)
                    n_images.append(a[None])
                n_images = torch.concatenate(n_images, axis=0)
                image = torch.mean(n_images, axis=0)
                image = image.permute(1,2,0).cpu().numpy()
                image = rgb2lab(image) # lab color space 0,1
                image = torch.tensor(image).permute(2,0,1).float()
            else:
                image = torch.mean(batch['image'],axis=0)
            image = logit(image) # mapping from [0,1] to [-inf,inf]
            self.albedo = torch.nn.Parameter(image[None])
            self.albedo.requires_grad = False
            break
    
    def initial_with_albedo(self, albedo):
       """
       expected albedo shape [1,3,H,W] as tensor
       """
       albedo = logit(albedo)
       self.albedo = torch.nn.Parameter(albedo)


    def get_albedo(self):
        """
        return albedo in scale [0,1]
        """
        return torch.sigmoid(self.albedo)
        
    def setup_normal_pipeline(self):
        self.pipe_normal = diffusers.MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
        ).to("cuda")

    def compute_normal(self, image):
        # @param image [0,1] tensor [3,h,3]
        # where X axis points right, Y axis points up, and Z axis points at the viewer
        # @see https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage
        if self.use_lab: # Normal model only accept RGB image
            device = image.device
            image = image.permute(1,2,0).cpu().numpy()
            image = lab2rgb(image)
            image = torch.tensor(image).to(device)
            image = image.permute(2,0,1)
        normals = self.pipe_normal(
            image,
            output_type='pt'
        )
        return normals

    def get_normal(self, image):
        if self.normal_map is not None:
            return self.normal_map 
        assert len(image.shape) == 3 and image.shape[0] == 3 # make sure it feed only 1 image
        normals = self.compute_normal(image)
        self.normal_map = normals.prediction[0]
        assert len(self.normal_map.shape) == 3 and self.normal_map.shape[0] == 3 #make sure self.normal_map is 3,H,W
        return self.normal_map

    def setup_albedo(self):
        """
        create the albedo tensor that can optimize shape [1,3,H,W]
        """
        self.albedo = torch.nn.Parameter(
            torch.randn(1,3, self.image_size[0], self.image_size[1]) * self.std_multiplier
        )
        self.albedo.requires_grad = False
    
    def setup_spherical_coefficient(self):
        """
        create the spherical coefficient tensor that can optimize shape [num_images, 3, 9]
        """
        initial_random = torch.randn(self.num_images, 3, 9) * self.std_multiplier
        initial_random[:,:,0] = initial_random[:,:,0] + (np.sqrt(4*np.pi)) # pass the image color
        self.shcoeffs = torch.nn.Parameter(
            initial_random
        )
        
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
            normal[:,0] * normal[:,1],          # X*Y
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

        # hard cap, shading should in range [0,1]
        #shading = torch.clamp(shading, 0, 1)
  

        if albedo is not None:
            # albedo range [0,1] * shading range [0,1] to image range [0,1]
            rendered = albedo * shading
        else:
            rendered = shading

        assert rendered.shape[1:] == normal.shape[1:] and shcoeffs.shape[0] == rendered.shape[0] # [bz, 3, h, w]
        return rendered
    
    def disable_albedo_optimization(self):
        self.optimize_albedo = False
        self.albedo.requires_grad = False

    def on_train_epoch_start(self):
        """Unfreeze after N epochs"""
        if self.current_epoch >= self.cold_start_albedo and self.optimize_albedo:
            self.albedo.requires_grad = True
        else:
            self.albedo.requires_grad = False



    def training_step(self, batch, batch_idx):
        bz = batch['image'].shape[0]
        assert bz == self.num_images # expect the batch size to be always same with num image to make thing easier 

        normal_map = self.get_normal(batch['image'][0])[None] #[1,3,H,W]

        render_image = self.render_image(self.shcoeffs, normal_map, self.get_albedo())

        render_image = torch.clamp(render_image, 0, 1) # hard constain image rance
        
        loss = torch.nn.functional.mse_loss(render_image, batch['image'])

        if self.sh_regularize > 0:
            loss += self.sh_regularize * (torch.norm(self.shcoeffs[...,1:], p=2) + torch.norm(self.shcoeffs[...,0] - (np.sqrt(4*np.pi))))  #add L2 regularize for stability
    
        if self.sh_3channel > 0:
            loss += self.sh_3channel * (self.shcoeffs.var(dim=1).mean()) # variation loss to keep 3 channel stay together.

        self.log('train/loss', loss)

        return loss

    def log_tensorboard(self, batch, batch_idx):
        # save normal ball
        bz = batch['image'].shape[0]
        normal_ball_front, mask = get_ideal_normal_ball_y_up(256)
        normal_ball_front =  torch.tensor(normal_ball_front).to(batch['image'].device).permute(2,0,1)[None]
        self.logger.experiment.add_image(
            'ball_front/normal',
            n2v(normal_ball_front[0]),
            self.global_step
        )
        normal_ball_back = normal_ball_front.clone()
        normal_ball_back[:,2] *= -1 
        self.logger.experiment.add_image(
            'ball_rear/normal',
            n2v(normal_ball_back[0]),
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
        viz_albedo = self.get_albedo()[0]
        if self.use_lab:
            viz_albedo = viz_lab(viz_albedo)
        self.logger.experiment.add_image(
            f'albedo',
            viz_albedo,
            self.global_step
        )
        #save albedo 
        try:
            log_dir = self.logger.log_dir
        except:
            log_dir = self.log_dir
        epoch_id = 0 if self.global_step == 0 else self.current_epoch + 1
        os.makedirs(f"{log_dir}/albedo", exist_ok=True)
        torchvision.utils.save_image(viz_albedo, f"{log_dir}/albedo/albedo_{epoch_id:04d}.png")
        if self.log_shading:
            # save frontside chromeball to visualize the lighting 
            ball_image_front = self.render_image(self.shcoeffs, normal_ball_front)
            for i in range(bz):
                viz_ball = ball_image_front[i]
                if self.use_lab:
                    viz_ball = viz_lab(viz_ball)
                self.logger.experiment.add_image(
                    f"ball_front/{batch['name'][i]}",
                    viz_ball,
                    self.global_step
                )
            # save backside chromeball to visualize the lighting 
            ball_image_back = self.render_image(self.shcoeffs, normal_ball_back)
            for i in range(bz):
                viz_ball = ball_image_back[i]
                if self.use_lab:
                    viz_ball = viz_lab(viz_ball)
                self.logger.experiment.add_image(
                    f"ball_rear/{batch['name'][i]}",
                    viz_ball,
                    self.global_step
                )
            # render shading
            render_shading = self.render_image(self.shcoeffs, normal_map[None])
            for i in range(bz):
                shading = render_shading[i].detach().cpu().permute(1,2,0).numpy()
                shading, _, _ = self.hdr2ldr(shading)
                shading = torch.tensor(shading).permute(2,0,1)

                self.logger.experiment.add_image(
                    f"shading/{batch['name'][i]}",
                    n2v(s2n(shading), use_lab=self.use_lab),
                    self.global_step
                )
        # render image
        render_image = self.render_image(self.shcoeffs, normal_map[None], self.get_albedo())
        if self.log_shading:
            for i in range(bz):
                cat_image = torch.concatenate([batch['image'][i:i+1],render_image[i:i+1]],axis=0)
                if self.use_lab:
                    cat_image[0] = viz_lab(cat_image[0])
                    cat_image[1] = viz_lab(cat_image[1])
                self.logger.experiment.add_image(
                    f"rendered/{batch['name'][i]}",
                    torchvision.utils.make_grid(cat_image),
                    self.global_step
                )
        mse = torch.nn.functional.mse_loss(render_image, batch['image'])
        psnr_value = 10 * torch.log10(1.0**2 / mse)
        self.log('val/render_psnr', psnr_value)
        self.log('val/loss', mse)
    
    def save_shcoeffs(self, path=None):
        if path is None: 
            try:
                log_dir = self.logger.log_dir
            except:
                log_dir = self.log_dir
            path = f"{log_dir}/shcoeffs.npy"
        np.save(path, self.shcoeffs.cpu().detach().numpy())

    def save_shading(self, path=None, image=None, file_type="png"):
        if file_type == "exr":
            import ezexr
        
        if path is None:
            try:
                log_dir = self.logger.log_dir
            except:
                log_dir = self.log_dir
        if image is not None:
            normal_map = self.get_normal(image)
        else:
            normal_map = self.normal_map
        render_shading = self.render_image(self.shcoeffs.to(normal_map.device), normal_map[None], albedo=None)
        shading_dir = os.path.join(log_dir, "shadings")
        os.makedirs(shading_dir, exist_ok=True)
        if self.use_lab:
            shading_lab_dir =  os.path.join(log_dir, "shadings_lab")
            os.makedirs(shading_lab_dir,exist_ok=True)

        for i in range(self.shcoeffs.shape[0]):
            if file_type == "exr":
                exr_dir = os.path.join(log_dir, "shadings_exr")
                os.makedirs(exr_dir, exist_ok=True)
                exr_path = os.path.join(exr_dir, f"dir_{i}_mip2.exr")
                ezexr.imwrite(render_shading[i],exr_path)
            else:
                if self.use_lab:
                    c_shading = render_shading[i]
                    c_shading = c_shading.permute(1,2,0).cpu().detach().numpy()
                    c_shading = skimage.img_as_ubyte(c_shading)
                    skimage.io.imsave(os.path.join(shading_lab_dir, f"dir_{i}_mip2.png"), c_shading)
                c_shading = render_shading[i].cpu().detach()
                c_shading = n2v(s2n(c_shading), use_lab=self.use_lab)
                c_shading = c_shading.permute(1,2,0).numpy()
                c_shading = skimage.img_as_ubyte(c_shading)
                skimage.io.imsave(os.path.join(shading_dir, f"dir_{i}_mip2.png"), c_shading)


    def save_render(self, path=None, image=None):
        if path is None:
            try:
                log_dir = self.logger.log_dir
            except:
                log_dir = self.log_dir
        if image is not None:
            normal_map = self.get_normal(image)
        else:
            normal_map = self.normal_map
        albedo = self.get_albedo().to(normal_map.device)
        render_shading = self.render_image(self.shcoeffs.to(normal_map.device), normal_map[None], albedo=albedo)
        shading_dir = os.path.join(log_dir, "render")
        os.makedirs(shading_dir, exist_ok=True)

        for i in range(self.shcoeffs.shape[0]):
            c_shading = render_shading[i].cpu().detach()
            c_shading = n2v(s2n(c_shading), use_lab=self.use_lab)
            c_shading = c_shading.permute(1,2,0).numpy()
            c_shading = skimage.img_as_ubyte(c_shading)
            skimage.io.imsave(os.path.join(shading_dir, f"dir_{i}_mip2.png"), c_shading)

            

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            try:
                for arg, value in vars(args).items():
                    self.logger.experiment.add_text(f'args/{arg}', str(value), self.global_step)
            except Exception as e:
                pass
        self.log_tensorboard(batch, batch_idx)

    def configure_optimizers(self):
        # change optimizer for least square BCG LM
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
    x,y = np.meshgrid(x,y)

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

def viz_lab(tensor):
    device = tensor.device
    tensor = tensor.permute(1,2,0).cpu().numpy()
    tensor = lab2rgb(tensor) # range [0,1]
    tensor = torch.tensor(tensor).to(device)
    tensor = tensor.permute(2,0,1)
    return tensor

def n2v(tensor, use_lab = False):
    # normalize (-1,1) to visualize [0,1]

    if use_lab:
        tensor = torch.clamp(tensor, -1, 1) # range [-1,1]
        device = tensor.device
        tensor = tensor.permute(1,2,0).cpu().numpy()
        tensor = lab2rgb(tensor) # range [0,1]
        tensor = torch.tensor(tensor).to(device)
        tensor = tensor.permute(2,0,1)
    else:
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor

def rgb2lab(img):
    """
    Convert RGB to LAB
    Parameters:
        img: np.array (range 0,1)
    Returns:
        img: np.array (range 0,1)
    """
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.shape[2] == 3 and len(img.shape) == 3

    img = skimage.color.rgb2lab(img)

    # Normalize L (0-100) → (0,1) and A/B (-128,127) → (-1,1)
    img[:,:,0] = img[:,:,0] / 100
    img[:,:,1] = (img[:,:,1] + 128) / 255.0
    img[:,:,2] = (img[:,:,2] + 128) / 255.0

    return img

def lab2rgb(img):
    """
    Convert LAB to RGB
    Parameters:
        img: np.array (range 0,1)
    Returns:
        img: np.array (range 0,1)

    """
    assert img.shape[2] == 3  and len(img.shape) == 3

    is_torch = torch.is_tensor(img)

    if is_torch:
        device = img.device
        img = img.permute(1,2,0).cpu().numpy()

    # Convert to LAB space from [0,1] to L (0-100), A/B (-128,127)
    img[:,:,0] = np.clip(img[:,:,0]* 100, 0, 100)   # L should be clipped
    img[:,:,1] = np.clip(img[:,:,1] * 255 - 128, -128, 127)  # A channel
    img[:,:,2] = np.clip(img[:,:,2] * 255 - 128, -128, 127)  # B channel

    # Convert LAB to RGB
    img = skimage.color.lab2rgb(img)

    if is_torch:
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).to(device)

    assert img.min() >= 0.0 and img.max() <= 1.0
    return img

def atanh(x, eps=1e-6):
    # Clamp x to avoid -1 or 1 exactly
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * torch.log((1 + x) / (1 - x))

def logit(x, eps=1e-6):
    """
    Compute the inverse sigmoid (logit function) safely.
    
    Args:
        x (torch.Tensor): Input tensor with values in range (0,1).
        eps (float): Small value to avoid log of zero or division by zero.
    
    Returns:
        torch.Tensor: Output tensor with values in range (-inf, inf).
    """
    x = torch.clamp(x, eps, 1 - eps)  # Avoid numerical issues at 0 and 1
    return torch.logit(x)

def s2n(a):
    return (a * 2.0) - 1.0

def main():

    SCENE_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom10"
    train_dataset = MultiIluminationSceneDataset(
        scene_path=SCENE_DIR,
        data_multiplier=args.dataset_multipiler,
        image_size=(512,512),
        use_lab = args.use_lab == 1
    )
    val_dataset = MultiIluminationSceneDataset(
        scene_path=SCENE_DIR,
        data_multiplier=1,
        image_size=(512,512),
        use_lab = args.use_lab == 1
    )
    model = AlbedoOptimization(
        num_images = train_dataset.get_num_images(),
        lr_albedo = args.lr_albedo,
        lr_shcoeff = args.lr_shcoeff,
        std_multiplier = args.std_multiplier,
        cold_start_albedo = args.cold_start_albedo,
        sh_regularize = args.sh_regularize,
        sh_3channel = args.sh_3channel,
        use_lab = args.use_lab == 1
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


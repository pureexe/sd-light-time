print("IMPORIING OS")
import os 
print("IMPORTING PIL")
from PIL import Image
print("IMPORTING tqdm")
from tqdm.auto import tqdm
print("IMPORTING numpy")
import numpy as np
print("IMPORTING torch")
import torch
import torch.nn.functional as F
print("IMPORTING pyshtools")
import pyshtools
print("IMPORTING EINOPS")
from einops import rearrange
print("IMPORTING NormalBAE")
from controlnet_aux import NormalBaeDetector
print("IMPORTING CONTROLNET_UTIL")
from controlnet_aux.util import HWC3, resize_image
print("IMPORTING torchvision")
import torchvision
import ezexr


ROOT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"
IMAGE_DIR = "images"
COEFF_DIR = "shcoeffs_order2_hdr"


def unfold_sh_coeff(flatted_coeff, max_sh_level=2):
    """
    flatten spherical harmonics coefficient to 3xC matrix
    #  array format [0_0, 1_-1, 1_0, 1_1, 2_-2, 2_-1, 2_0, 2_1, 2_2]
    """
    sh_coeff = np.zeros((3, 2, max_sh_level+1, max_sh_level+1))
    for i in range(3):
        c = 0
        for j in range(max_sh_level+1):
            for k in range(j, 0, -1):
                sh_coeff[i, 1, j, k] = flatted_coeff[i, c]
                c +=1
            for k in range(j+1):
                sh_coeff[i, 0, j, k] = flatted_coeff[i, c]
                c += 1
    return sh_coeff

def get_environment_map(
        sh, lmax=2,
        image_width=512
    ):
    # Generate random spherical harmonic coefficients
    loaded_coeff = sh
    # make sure that shcoeff is already unfold 
    assert len(sh.shape) == 4 and sh.shape[0] == 3 and sh.shape[1] == 2 # shape should be [3,2,L+1,L+1]
    
    output_image = []
    for ch in (range(3)):
        coeffs = loaded_coeff[ch]
            
        # Create SHCoeffs class object from the coefficients
        sh_coeffs = pyshtools.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)

        theta = np.linspace(np.pi / 2, -np.pi / 2, image_width)
        phi = np.linspace(0, np.pi * 2, 2*image_width)

        lat, lon = np.meshgrid(theta, phi, indexing='ij')

        # Evaluate the spherical harmonics on the grid
        grid_data = sh_coeffs.expand(grid="GLQ", lat=lat, lon=lon, lmax_calc=lmax, degrees=False)
        output_image.append(grid_data[...,None])    

    output_image = np.concatenate(output_image,axis=-1)
    return output_image

def get_light_vector(normal_map, viewing_vector=None):
    """
    Calculate the light vector L using the normal map N and viewing vector V.

    Args:
        normal_map (torch.Tensor): Tensor of shape [Batch, 3, H, W] representing the normal map.
        viewing_vector (torch.Tensor, optional): Tensor of shape [3] representing the viewing vector.
                                         Defaults to [0, 0, 1].

    Returns:
        torch.Tensor: Light vector tensor of shape [Batch, 3, H, W].
    """
    if viewing_vector is None:
        viewing_vector = torch.tensor([0.0, 0.0, 1.0], dtype=normal_map.dtype, device=normal_map.device)

    # Ensure viewing_vector is a column vector for broadcasting
    viewing_vector = viewing_vector.view(1, 3, 1, 1)  # [1, 3, 1, 1] for batch broadcasting

    # Normalize the normal map and viewing vector
    normal_map = normal_map / torch.norm(normal_map, dim=1, keepdim=True).clamp(min=1e-8)
    viewing_vector = viewing_vector / torch.norm(viewing_vector, dim=1, keepdim=True).clamp(min=1e-8)

    # Compute the reflection vector R = 2 * (N . V) * N - V
    dot_product = torch.sum(normal_map * viewing_vector, dim=1, keepdim=True)  # N . V
    reflection_vector = 2 * dot_product * normal_map - viewing_vector

    # Normalize the reflection vector to get the light vector
    light_vector = reflection_vector / torch.norm(reflection_vector, dim=1, keepdim=True).clamp(min=1e-8)

    return light_vector


def lambertian_render(normal_map, albedo, env_map, light_dir):
    """
    Perform Lambertian rendering given a normal map, albedo, equirectangular environment map,
    and light direction.

    Args:
        normal_map (torch.Tensor): Normal map of shape [batch, 3, H, W].
        albedo (torch.Tensor): Albedo map of shape [batch, 3, H, W].
        env_map (torch.Tensor): Equirectangular environment map of shape [batch, 3, A, 2A].
        light_dir (torch.Tensor): Light direction map of shape [batch, 3, H, W].

    Returns:
        torch.Tensor: Rendered image of shape [batch, 3, H, W].
    """
    # Ensure the inputs are valid
    batch, _, H, W = normal_map.shape
    _, _, A, _ = env_map.shape
    
    # Normalize the normal map and light direction to unit vectors
    normal_map = F.normalize(normal_map, dim=1)
    light_dir = F.normalize(light_dir, dim=1)

    # Compute dot product of normal_map and light_dir
    dot_products = torch.sum(normal_map * light_dir, dim=1, keepdim=True).clamp(min=0)  # [batch, 1, H, W]

    # Convert light direction to spherical coordinates for environment map lookup
    x, y, z = light_dir[:, 0], light_dir[:, 1], light_dir[:, 2]
    theta = torch.acos(z.clamp(-1, 1))  # [batch, H, W], latitude
    phi = torch.atan2(y, x)  # [batch, H, W], longitude
    phi = (phi + 2 * torch.pi) % (2 * torch.pi)  # Ensure phi is in [0, 2*pi]

    # Map spherical coordinates to normalized coordinates for grid sampling
    u = phi / (2 * torch.pi) * 2 - 1  # Longitude to [-1, 1] shape[1,512,512]
    v = theta / torch.pi * 2 - 1  # Latitude to [-1, 1] shape[1,512,512]

    # Create grid for grid_sample
    grid = torch.stack((u, v), dim=-1)  # [batch, H, W, 2]
    
    #.permute(0, 2, 3, 1)  # [batch, H, W, 2]
    # torch.Size([1, 512, 2, 512])

    # Use grid_sample to sample the environment map
    env_map_sampled = F.grid_sample(env_map, grid, mode='bilinear', align_corners=False)  # [batch, 3, H, W]

    # Compute irradiance from dot product and sampled environment map
    irradiance = dot_products * env_map_sampled  # [batch, 3, H, W]

    # Compute final rendered image
    rendered_image = irradiance * albedo

    return rendered_image

def get_queues():
    #scenes = sorted(os.listdir(os.path.join(ROOT_DIR, IMAGE_DIR)))
    scenes = ['14n_copyroom1','14n_copyroom10','14n_copyroom8']
    queues  = []
    for scene in scenes:
        for idx in range(25):
            queues.append((scene,idx))
    return queues

class NormalBaeDetectorPT(NormalBaeDetector):
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().to(device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)
            normal = self.model(image_normal)
            normal = normal[0][-1][:, :3]
            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()

        return normal

def main():
    print("GETTING QUEUE")
    queues = get_queues()
    preprocessor = NormalBaeDetectorPT.from_pretrained("lllyasviel/Annotators")
    preprocessor = preprocessor.to('cuda')
    
    #queues = queues[:25]
    pbar = tqdm(queues)
    pbar.set_description(f"")
    for info in pbar:
        pbar.set_postfix(item=f"{info[0]}/{info[1]}")
        idx = info[1]
        scene = info[0]
        image = Image.open(f"{ROOT_DIR}/{IMAGE_DIR}/{scene}/dir_{idx}_mip2.jpg").convert("RGB")
        normal_map = preprocessor(image, output_type="pt") 
        normal_map = torch.from_numpy(normal_map)
        normal_map = normal_map.permute(2,0,1)[None]
        # normal map shape [1,3,512,512] range (-1,1)
        # load shcoeff 
        shcoeff = np.load(f"{ROOT_DIR}/{COEFF_DIR}/{scene}/dir_{idx}_mip2.npy") # shcoeff shape (3,9)
        shcoeff = unfold_sh_coeff(shcoeff) #shape [3,2,3,3]
        env_map = get_environment_map(shcoeff) # numpy shape (512, 1024, 3)
        env_map = torch.from_numpy(env_map).permute(2,0,1)[None].float() #env_map.shape [3,512,1024]
        light_dir = get_light_vector(normal_map)
        albedo = torch.ones_like(normal_map)

        rgb_map = lambertian_render(normal_map, albedo, env_map, light_dir)
        image = rgb_map[0]       #image = torchvision.transforms.functional.to_pil_image(rgb_map[0])
        image = image.permute(1,2,0).numpy()
        os.makedirs(os.path.join("output", "hdr", scene),exist_ok=True)
        output_path = os.path.join("output", "hdr",  scene, f"dir_{idx}_mip2.exr")
        ezexr.imwrite(output_path, image)
    

if __name__ == "__main__":
    main()
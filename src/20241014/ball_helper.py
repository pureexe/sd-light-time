import torch 
import numpy as np 
import skimage
import os
from transformers import pipeline as hf_pipeline
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from PIL import Image


def create_circle_tensor(num_pixel, circle_size):
    """
    Create a PyTorch tensor with a circle in the middle.
    
    Args:
        num_pixel (int): The size of the tensor (num_pixel x num_pixel).
        circle_size (int): The diameter of the circle.
        
    Returns:
        torch.Tensor: A tensor with a circle in the middle (1.0 inside, 0.0 outside). shape [num_pixel, num_pixel]
    """
    # Create a tensor of zeros
    tensor = torch.zeros((num_pixel, num_pixel), dtype=torch.float32)
    
    # Compute the center and radius of the circle
    center = num_pixel // 2
    radius = circle_size // 2
    
    # Define the grid
    y, x = torch.meshgrid(torch.arange(num_pixel), torch.arange(num_pixel))
    
    # Calculate the distance from the center
    distance_from_center = torch.sqrt((x - center)**2 + (y - center)**2)
    
    # Update tensor values based on the distance from the center
    tensor[distance_from_center <= radius] = 1.0
    
    return tensor

def get_reflection_vector_map(I: np.array, N: np.array):
    """
    UNIT-TESTED
    Args:
        I (np.array): Incoming light direction #[None,None,3]
        N (np.array): Normal map #[H,W,3]
    @return
        R (np.array): Reflection vector map #[H,W,3]
    """
    
    # R = I - 2((I⋅ N)N) #https://math.stackexchange.com/a/13263
    dot_product = (I[...,None,:] @ N[...,None])[...,0]
    R = I - 2 * dot_product * N
    return R


def cartesian_to_spherical(cartesian_coordinates):
    """Converts Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coordinates: A NumPy array of shape [..., 3], where each row
        represents a Cartesian coordinate (x, y, z).

    Returns:
        A NumPy array of shape [..., 3], where each row represents a spherical
        coordinate (r, theta, phi).
    """

    x, y, z = cartesian_coordinates[..., 0], cartesian_coordinates[..., 1], cartesian_coordinates[..., 2]
    r = np.linalg.norm(cartesian_coordinates, axis=-1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return np.stack([r, theta, phi], axis=-1)


def get_ideal_normal_ball(size):
    
    """
    UNIT-TESTED
    BLENDER CONVENTION
    X: forward
    Y: right
    Z: up
    
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(1, -1, size)
    
    #use indexing 'xy' torch match vision's homework 3
    y,z = torch.meshgrid(y, z ,indexing='xy') 
    
    x = (1 - y**2 - z**2)
    mask = x >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    x = torch.sqrt(x)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask


def envmap2chromeball(env_map):
    """ convert envmap to image with ball in the middle

    """

    assert env_map.shape[1] == 3

    normal_ball, _ = get_ideal_normal_ball(512)

    # verify that x of normal is in range [0,1]
    assert normal_ball[:,:,0].min() >= 0 
    assert normal_ball[:,:,0].max() <= 1 
    
    # camera is pointing to the ball, assume that camera is othographic as it placing far-away from the ball
    I = np.array([1, 0, 0]) 
        
    ball_image = np.zeros_like(normal_ball)
    
    reflected_rays = get_reflection_vector_map(I[None,None], normal_ball)
    spherical_coords = cartesian_to_spherical(reflected_rays)
    
    theta_phi = spherical_coords[...,1:]
    
    # scale to [0, 1]
    # theta is in range [-pi, pi],
    theta_phi[...,0] = (theta_phi[...,0] + np.pi) / (np.pi * 2)
    # phi is in range [0,pi] 
    theta_phi[...,1] = theta_phi[...,1] / np.pi
    
    # mirror environment map because it from inside to outside
    theta_phi = 1.0 - theta_phi
    
    with torch.no_grad():
        # convert to torch to use grid_sample
        theta_phi = torch.from_numpy(theta_phi[None])
        env_map = torch.from_numpy(env_map[None]).permute(0,3,1,2)
        # grid sample use [-1,1] range
        grid = (theta_phi * 2.0) - 1.0
        ball_image = torch.nn.functional.grid_sample(env_map.float(), grid.float(), mode='bilinear', padding_mode='border', align_corners=True)
    return ball_image, normal_ball

def pipeline2controlnetinpaint(pipe, controlnet = None):
    # convert any stable diffusion pipeline to control inpainting pipeline
    new_pipe = StableDiffusionControlNetInpaintPipeline(
        vae = pipe.vae,
        text_encoder = pipe.text_encoder,
        tokenizer = pipe.tokenizer,
        unet = pipe.unet,
        controlnet = controlnet if controlnet is not None else pipe.controlnet,
        scheduler = pipe.scheduler,
        safety_checker = pipe.safety_checker,
        feature_extractor = pipe.feature_extractor,
        requires_safety_checker = False,
    )
    return new_pipe 

def ball_center_crop(image, ball_size = 128):
    H, W = image.size
    left = (W - ball_size) // 2
    top = (H - ball_size) // 2
    right = left + ball_size
    bottom = top + ball_size
    cropped_image = image.crop((left, top, right, bottom))
    # apply mask 
    mask = create_circle_tensor(128, 128).cpu().numpy()[...,None]
    cropped_image = np.array(cropped_image)  / 255.0 * mask
    cropped_image = (cropped_image * 255).clip(0, 255).astype(np.uint8)
    cropped_image = Image.fromarray(cropped_image)
    return cropped_image

def inpaint_chromeball(image, pipe=None, inpainting_mask = None, condition_mask = None):
    """
    inpainting EV 0 chromeball 

    Args:
        pipe (torch.nn.Module): Inpainting model
        image (PIL.Image): Chromeball image [B,3,512,512]
        inpainting_mask (PIL.Image, optional): Inpainted mask [512,512]. Defaults to None.
        condition_mask (PIL.Image, optional): Condition mask [512,512]. Defaults to None.
    
    Returns:
        PIL.Image: Inpainted chromeball image [B,3,512,512]
    """
    if inpainting_mask is None:
        inpainting_mask = tensor_to_pil(create_circle_tensor(512, 138))
    
    if condition_mask is None:
        condition_mask = tensor_to_pil(create_circle_tensor(512, 132))

    if pipe is None:
        # load sd inpaint pipeline
        sd_path="runwayml/stable-diffusion-v1-5"
        controlnet_path="lllyasviel/sd-controlnet-depth"
        MASTER_TYPE = torch.float16
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            sd_path,
            controlnet=ControlNetModel.from_pretrained(controlnet_path, torch_dtype=MASTER_TYPE),
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        ).to('cuda')

    PROMPT = "a perfect mirrored reflective chrome ball sphere"
    NEG_PROMPT = "matte, diffuse, flat, dull"
    output_image = pipe(
        prompt = PROMPT,
        image = image,
        mask_image = inpainting_mask,
        control_image = condition_mask,
        strength = 1.0,
        num_inference_steps = 50,
        guidance_scale = 5.0,
        negative_prompt= NEG_PROMPT
    )['images']

    if not isinstance(image, list):
        output_image = output_image[0]

    return output_image

def tensor_to_pil(tensor):
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    assert tensor.dim() == 4
    assert tensor.min() >= 0 and tensor.max() <= 1    
    img = tensor.permute(0, 2, 3, 1).cpu().numpy()[0]
    if img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=2)
    img = Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8))
    return img

def get_depth_estimator(model="Intel/dpt-hybrid-midas", device="cuda"):
    return hf_pipeline("depth-estimation", model=model, device=device)

def depth_prediction(image, depth_estimator = None):
    """_summary_

    Args:
        image (PIL.Image/list(PIL.Image)): image for compute depth in PIL format
        depth_estimator (_type_, optional): _description_. Defaults to None.

    Returns:
        PIL.Image/list(PIL.Image): depth map in PIL format 3 channels
    """
    if depth_estimator is None:
        depth_estimator = get_depth_estimator()

    return_list = False
    if not isinstance(image, list):
        image = [image]
    else:
        return_list = True

    image_out = []
    for img in image:
        depth_map = depth_estimator(img)['predicted_depth']
        W, H = img.size
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        img = torch.cat([depth_map] * 3, dim=1)
        img = tensor_to_pil(img)
        image_out.append(img)
    if return_list:
        return image_out
    else:
        return image_out[0]


def add_ball_middle(image, ball_mask):
    """_summary_

    Args:
        image (PIL.Image): image to add ball in the middle
        ball_mask (np.array): ball image [1,512,512]

    Returns:
        PIL.Image: image with ball in the middle
    """
    image = np.array(image) / 255.0
    ball_mask = np.array(ball_mask) / 255.0
    if ball_mask.ndim == 2:
        ball_mask = ball_mask[:, :, None]

    # if inside the mask, become white elese keep the original image
    masked_image = (1.0 - ball_mask) * image + ball_mask
    masked_image = (masked_image * 255).clip(0, 255).astype(np.uint8)   
    masked_image = Image.fromarray(masked_image)
    return masked_image
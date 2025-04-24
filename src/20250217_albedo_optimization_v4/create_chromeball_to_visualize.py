import torch
import diffusers
import numpy as np
import os 
from PIL import Image 
import torchvision
import argparse 
import skimage 
from tonemapper import TonemapHDR

import ezexr 

import numpy as np
from scipy.spatial.transform import Rotation as R
import pyshtools

MASTER_TYPE = torch.float32


##################
# def rotate_sh2(sh_coeffs, yaw, pitch):
#     """
#     Rotate 2nd order spherical harmonic coefficients by given yaw (phi) and pitch (theta).
    
#     Args:
#         sh_coeffs: numpy array of shape [9, 3] (3 channels of SH coefficients).
#         yaw: rotation around Y-axis (left/right), in radians.
#         pitch: rotation around X-axis (up/down), in radians.

#     Returns:
#         Rotated SH coefficients of shape [9,3].
#     """
#     assert sh_coeffs.shape == (9, 3), "Input SH coefficients must have shape [9,3]"

#     # Convert yaw (horizontal) and pitch (vertical) into a full 3D rotation matrix
#     R_mat = R.from_euler('YX', [yaw, pitch], degrees=False).as_matrix()

#     # Compute Wigner-d matrices for l=1 and l=2
#     dj_l1 = pyshtools.rotate.djpi2(1)  # Wigner-d for l=1
#     dj_l2 = pyshtools.rotate.djpi2(2)  # Wigner-d for l=2

#     # Compute SH rotation matrices for l=1 (3x3) and l=2 (5x5)
#     R_l1 = pyshtools.rotate.SHRotateRealCoef(np.eye(3), R_mat, dj_l1)  # l=1
#     R_l2 = pyshtools.rotate.SHRotateRealCoef(np.eye(5), R_mat, dj_l2)  # l=2

#     # Apply rotation to SH coefficients
#     rotated_sh = np.zeros_like(sh_coeffs)

#     # l=0 (DC term, remains unchanged)
#     rotated_sh[0, :] = sh_coeffs[0, :]

#     # l=1 (Apply 3x3 rotation)
#     rotated_sh[1:4, :] = R_l1 @ sh_coeffs[1:4, :]

#     # l=2 (Apply 5x5 rotation)
#     rotated_sh[4:9, :] = R_l2 @ sh_coeffs[4:9, :]

#     return rotated_sh

def legendre_rotation(l, m1, m2, rotation_matrix):
    """
    Compute the rotation coefficient for spherical harmonics using 
    an approximation based on the standard rotation matrix.

    Parameters:
    - l: int, the SH band (only l=2 is used here)
    - m1, m2: int, SH index (-l to l)
    - rotation_matrix: (3,3) numpy array, standard rotation matrix

    Returns:
    - Rotation coefficient for SH basis transformation
    """
    # Approximate rotation using dot product
    if abs(m1 - m2) > 2:
        return 0  # Rotation is insignificant for distant indices

    # Use the corresponding elements of the 3x3 rotation matrix
    if m1 == m2:
        return rotation_matrix[0, 0]  # Approximate diagonal rotation
    elif abs(m1 - m2) == 1:
        return rotation_matrix[0, 1]  # Off-diagonal
    elif abs(m1 - m2) == 2:
        return rotation_matrix[0, 2]  # Smallest contribution

    return 0  # Default fallback


def rotate_sh_coeffs(sh_coeffs, phi, theta):
    """
    Rotate spherical harmonic coefficients (order 2) given angles phi and theta.

    Parameters:
    - sh_coeffs: numpy array of shape (9, 3), SH coefficients (order 2, 3 channels)
    - phi: float, rotation in horizontal direction (-π to π)
    - theta: float, rotation in vertical direction (-π/2 to π/2)

    Returns:
    - rotated_sh: numpy array of shape (9, 3), rotated SH coefficients
    """
    assert sh_coeffs.shape == (9, 3), "Input SH coefficients must have shape [9,3]"
    # Convert rotation angles into a 3x3 rotation matrix
    rotation_matrix = R.from_euler('YX', [phi, theta], degrees=False).as_matrix()

    # Initialize a 9x9 identity rotation matrix
    R_sh = np.eye(9)

    # Apply the 3x3 rotation for l=1
    R_sh[1:4, 1:4] = rotation_matrix  # Standard vector rotation

    # Compute the 5x5 rotation for l=2 (more complex)
    l2_rotation = np.zeros((5, 5))
    m_values = [-2, -1, 0, 1, 2]

    for i, mi in enumerate(m_values):
        for j, mj in enumerate(m_values):
            # Compute the rotation effect for each pair (mi, mj)
            l2_rotation[i, j] = legendre_rotation(2, mi, mj, rotation_matrix)

    # Insert the l=2 rotation into R_sh
    R_sh[4:9, 4:9] = l2_rotation

    # Apply the rotation to all 3 channels
    rotated_sh = np.dot(R_sh, sh_coeffs)

    return rotated_sh



##################

class ChromeballRenderer():
    def __init__(self):
        self.setup_spherical_coefficient()
        self.hdr2ldr = TonemapHDR(gamma=1.0, percentile=50, max_mapping=0.5)
        self.setup_normal_pipeline()
        self.albedo = None
        
    def set_albedo(self, albedo):
        self.albedo = albedo

    def setup_normal_pipeline(self):
        self.pipe_normal = diffusers.MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
        ).to("cuda")
        self.normal_map = None
    
    def get_normal(self, image):
        if self.normal_map is not None:
            return self.normal_map 
        assert len(image.shape) == 3 and image.shape[0] == 3 # make sure it feed only 1 image
        normals = self.compute_normal(image)
        self.normal_map = normals.prediction[0]
        assert len(self.normal_map.shape) == 3 and self.normal_map.shape[0] == 3 #make sure self.normal_map is 3,H,W
        return self.normal_map

    def compute_normal(self, image):
        # @param image [0,1] tensor [3,h,3]
        # where X axis points right, Y axis points up, and Z axis points at the viewer
        # @see https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage
        normals = self.pipe_normal(
            image,
            output_type='pt'
        )
        return normals
    
    def tonemap(self, render_shading):
        """
        Convert from torch to numpy to do tonemap and the nconvert back to torch
        """
        assert len(render_shading.shape) == 3
        device = render_shading.device
        shading = render_shading.detach().cpu().permute(1,2,0).numpy()
        shading, _, _ = self.hdr2ldr(shading)
        shading = torch.tensor(shading).permute(2,0,1)
        return shading
    
    def setup_spherical_coefficient(self):
        """
        create the spherical coefficient tensor that can optimize shape [num_images, 3, 9]
        """
        self.num_images = 25
        self.std_multiplier = 1e-4
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
    
    def load_shcoeffs(self, npy_path):
        shcoeff = np.load(npy_path)
        shcoeff = torch.tensor(shcoeff)
        self.shcoeffs = torch.nn.Parameter(
            shcoeff
        )
    
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

        if albedo is not None:
            # albedo range [0,1] * shading range [0,1] to image range [0,1]
            rendered = albedo * shading
        else:
            rendered = shading

        assert rendered.shape[1:] == normal.shape[1:] and shcoeffs.shape[0] == rendered.shape[0] # [bz, 3, h, w]
        return rendered

    def render_chromeball(self):
        normal_ball_front, mask = get_ideal_normal_ball_y_up(512)
        normal_ball_front =  torch.tensor(normal_ball_front).permute(2,0,1)[None]
        ball_image_front = self.render_image(self.shcoeffs, normal_ball_front)
        for i in range(25):
            #ball_image_map = self.tonemap(ball_image_front[i])
            ball_image_map = torch.clamp(ball_image_front[i], 0, 1)
            ball_image_map = ball_image_map.cpu().permute(1,2,0).numpy()
            ball_image_map = ball_image_map * mask[...,None]
            out_img = skimage.img_as_ubyte(ball_image_map)
            skimage.io.imsave(f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250217_albedo_optimization_v4/output/render_ball_test/ball/{i:02d}.png",out_img)
            
    def render_rotated_chromeball(self, chromeball_dir, light_id):
        # create directory 
        os.makedirs(chromeball_dir, exist_ok=True)
        normal_ball_front, mask = get_ideal_normal_ball_y_up(512)
        normal_ball_front =  torch.tensor(normal_ball_front).permute(2,0,1)[None]
        sh_coeffs = self.shcoeffs[light_id:light_id+1]
        for i in range(60):
            new_shcoeff = sh_coeffs.clone()[0].permute(1,0).numpy()
            step = i / 60.0 * np.pi * 2
            new_shcoeff = rotate_sh_coeffs(new_shcoeff, step, 0)
            new_shcoeff = torch.tensor(new_shcoeff).permute(1,0)[None]
            ball_image_front = self.render_image(new_shcoeff, normal_ball_front)
            #ball_image_map = torch.clamp(ball_image_front[0], 0, 1)
            ball_image_map = ball_image_front[0] / ball_image_front[0].max()
            ball_image_map = ball_image_map.cpu().permute(1,2,0).numpy()
            ball_image_map = ball_image_map * mask[...,None]
            out_img = skimage.img_as_ubyte(ball_image_map)
            skimage.io.imsave(os.path.join(chromeball_dir, f"{i:02d}.png"), out_img)

            
    def render_scene(self, shading_dir, shading_norm_dir, render_dir,light_id):
        normal = self.normal_map.cpu()[None]
        sh_coeffs = self.shcoeffs[light_id:light_id+1]
        
        # create directory
        os.makedirs(shading_dir, exist_ok=True)
        os.makedirs(shading_norm_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)
        
        for i in range(60):
            new_shcoeff = sh_coeffs.clone()[0].permute(1,0).numpy()
            step = i / 60.0 * np.pi * 2
            new_shcoeff = rotate_sh_coeffs(new_shcoeff, step, 0)
            new_shcoeff = torch.tensor(new_shcoeff).permute(1,0)[None]
            shading_image = self.render_image(new_shcoeff, normal)
            out_exr = shading_image[0].permute(1,2,0).cpu().numpy()
 
            ezexr.imwrite(os.path.join(shading_dir, f"dir_{i}_mip2.exr"), out_exr)
            
            if self.albedo is not None:
                rendered_image = self.albedo.cpu() * shading_image
                rendered_image = rendered_image[0].permute(1,2,0).numpy()
                rendered_image = np.clip(rendered_image, 0,1)
                rendered_image = skimage.img_as_ubyte(rendered_image)
                skimage.io.imsave(os.path.join(render_dir, f"dir_{i}_mip2.png"),rendered_image)
            
            ball_image_map = shading_image[0] / shading_image[0].max()
            #ball_image_map = torch.clamp(ball_image_front[0], 0, 1)
            ball_image_map = ball_image_map.cpu().permute(1,2,0).numpy()
            out_img = skimage.img_as_ubyte(ball_image_map)
            skimage.io.imsave(os.path.join(shading_norm_dir, f"dir_{i}_mip2.png"),out_img)
            
            
  
        
###################################

        
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

@torch.inference_mode()
def main():
    sh_path = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/shcoeff_order2_from_fitting/14n_copyroom10.npy"
    image_path = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom10/dir_0_mip2.jpg"
    renderer = ChromeballRenderer()
    renderer.load_shcoeffs(sh_path)
    renderer.render_rotated_chromeball()
    #image = skimage.io.imread(image_path)
    #image = skimage.img_as_float(image)
    #image = torch.tensor(image).permute(2,0,1).to('cuda')
    #normal = renderer.get_normal(image)
    #renderer.render_scene()
    

if __name__ == "__main__":
    main()
import torch
import numpy as np

"""
these function is taken directly from Diffusion Renderer
"""

def rgb2srgb(rgb):
    # mapling from linear rgb to sRGB
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1/2.4) - 0.055)

def reinhard(x, max_point=16):
    # Diffusion Renderer use reinhard tone mapping with max_point=16
    y_rein = x * (1 + x / (max_point ** 2)) / (1 + x)
    return y_rein

def hdr2log(env_hdr, log_scale=10000):
    # convert HDR to log space
    return torch.log1p(env_hdr) / np.log1p(log_scale)

def envmap_vec(res, device=None):
    return -latlong_vec(res, device).flip(0).flip(1) #[H, W, 3]

def latlong_vec(res, device=None):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    dir_vec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    # return dr.texture(cubemap[None, ...], dir_vec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]
    return dir_vec #[H, W, 3]
import os 
import torch 
import numpy as np
from PIL import Image
from multiprocessing import Pool
from LineNotify import notify
from tqdm.auto import tqdm

def applySHlight(normal_images, sh_coeff):
  N = normal_images
  sh = torch.stack(
    [
      N[0] * 0.0 + 1.0,
      N[0],
      N[1],
      N[2],
      N[0] * N[1],
      N[0] * N[2],
      N[1] * N[2],
      N[0] ** 2 - N[1] ** 2,
      3 * (N[2] ** 2) - 1,
    ],
    0,
  )  # [9, h, w]
  pi = np.pi
  constant_factor = torch.tensor(
    [
      1 / np.sqrt(4 * pi), #confirmed
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
    ]
  ).float()
  sh = sh * constant_factor[:, None, None]

  shading = torch.sum(
    sh_coeff[:, :, None, None] * sh[:, None, :, :], 0
  )  # [9, 3, h, w]

  return shading

def genSurfaceNormals(n):
  x = torch.linspace(-1, 1, n)
  y = torch.linspace(1, -1, n)
  y, x = torch.meshgrid(y, x)

  z = (1 - x ** 2 - y ** 2)
  mask = z < 0
  z[mask] = 0
  z = torch.sqrt(z)
  return torch.stack([x, y, z], 0), mask

def applySHlightXYZ(xyz, sh):
  out = applySHlight(xyz, sh)
  return out 

def drawSphere(sh, img_size=256, is_back=False, white_bg=False):
  n = img_size
  xyz, mask = genSurfaceNormals(n)
  if(is_back):
    xyz[2] = xyz[2] * -1
  if white_bg:
    xyz[:, mask] = 1
  out = applySHlightXYZ(xyz, sh)
  out[:, xyz[2] == 0] = 0
  return out

def process_image(idx):
    subdir_id = idx // 1000
    light = np.load(f"datasets/face/face2000_single/light/{subdir_id*1000:05d}/{idx:05d}_light.npy")
    sh_viz = drawSphere(torch.tensor(light)).permute(1,2,0).numpy()
    sh_gray = 0.299*sh_viz[..., 0]  + 0.587*sh_viz[..., 1] + 0.114*sh_viz[..., 2]
    #highest_position = np.unravel_index(np.argmax(sh_gray), sh_gray.shape)
    highest_position = np.unravel_index(np.argmin(sh_gray), sh_gray.shape)
    #np.save(f"datasets/face/face2000_single_viz/ball_front_fullscale/{idx:05d}.npy",sh_viz)
    return highest_position

@notify
def main():
    ball_image = np.zeros((256,256,3))
    highest_positions = []
    with Pool(32) as p:
        highest_positions = list(tqdm(p.imap(process_image, range(2000), chunksize=1), total=2000))
    for pos in highest_positions:
        ball_image[pos[0],pos[1],0] = 1
        ball_image[pos[0],pos[1],2] = 1
    img = Image.fromarray(np.uint8(ball_image*255))
    img.save("datasets/face/face2000_single_viz/ball_front_min.png")

  
if __name__ == "__main__":
  main()
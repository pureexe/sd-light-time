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

ROOT_DIR = "datasets/face/face70000"
INPUT_DIR = f"{ROOT_DIR}/face_light"
OUTPUT_DIR = f"{ROOT_DIR}/ball"

def process_sh(meta):
    sh_coeff, out_path = meta
    sh_viz = drawSphere(torch.tensor(sh_coeff)).permute(1,2,0).numpy()
    sh_gray = 0.299*sh_viz[..., 0]  + 0.587*sh_viz[..., 1] + 0.114*sh_viz[..., 2]
    highest_position = np.unravel_index(np.argmax(sh_gray), sh_gray.shape)
    DOT_SIZE = 5
    pos_top = highest_position[0] - DOT_SIZE
    pos_bottom = highest_position[0] + DOT_SIZE
    pos_left = highest_position[1] - DOT_SIZE
    pos_right = highest_position[1] + DOT_SIZE
    # clamp value in range [0,255]
    pos_top = max(0, pos_top)
    pos_bottom = min(255, pos_bottom)
    pos_left = max(0, pos_left)
    pos_right = min(255, pos_right)
    sh_viz = np.clip((sh_viz * 0.7)*  255, 0, 255)
    sh_viz[pos_top:pos_bottom, pos_left:pos_right] = [255,  0, 0]
    im = Image.fromarray(sh_viz.astype(np.uint8))
    im.save(out_path)

def process_image(file_path):
    out_path = os.path.join(OUTPUT_DIR, file_path.replace("_light.npy", ".png"))
    if os.path.exists(out_path):
        return
    light = np.load(os.path.join(INPUT_DIR, file_path))
    meta = (light, out_path)
    process_sh(meta)
    

# @notify
# def main():
#     # create output directory
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # list all file that end with _light.npy in the input directory
#     files = sorted(os.listdir(INPUT_DIR))
#     EXT = "_light.npy"
#     files = [f for f in files if f.endswith(EXT)]


#     with Pool(32) as p:
#         highest_positions = list(tqdm(p.imap(process_image, files, chunksize=1), total=len(files)))



    

@notify
def main():
    # create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # read sh variable 
    with open('/pure/tu150/datasets/relight/face/ffhq/ffhq-train-light-anno.txt')as f:
        sh_raw = f.readlines()

    with open('/pure/tu150/datasets/relight/face/ffhq/ffhq-valid-light-anno.txt')as f:
        sh_raw += f.readlines()

    sh_dict = {}
    for line in tqdm(sh_raw):
        line = line.strip()
        components = line.split(' ')
        name = int(components[0].split('.')[0])
        sh = np.array([float(x) for x in components[1:]])
        sh = sh.reshape(9, 3)
        sh_dict[name] = sh
    sh_list = []
    for key in range(70000):
        os.makedirs(f"{OUTPUT_DIR}/{key // 1000 * 1000:05d}", exist_ok=True)
        meta = (sh_dict[key], f"{OUTPUT_DIR}/{key // 1000 * 1000:05d}/{key:05d}.png")
        sh_list.append(meta)

    with Pool(32) as p:
        highest_positions = list(tqdm(p.imap(process_sh, sh_list, chunksize=1), total=len(sh_list)))


    
  
if __name__ == "__main__":
  main()
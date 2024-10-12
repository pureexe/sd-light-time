
import numpy as np
import torch as pt

def applySHlight(normal_images, sh_coeff):
  N = normal_images
  sh = pt.stack(
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
  constant_factor = pt.tensor(
    [
      1 / np.sqrt(4 * pi),
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

  shading = pt.sum(
    sh_coeff[:, :, None, None] * sh[:, None, :, :], 0
  )  # [9, 3, h, w]

  return shading

def genSurfaceNormals(n):
  x = pt.linspace(-1, 1, n)
  y = pt.linspace(1, -1, n)
  y, x = pt.meshgrid(y, x)

  z = (1 - x ** 2 - y ** 2)
  mask = z < 0
  z[mask] = 0
  z = pt.sqrt(z)
  return pt.stack([x, y, z], 0), mask

def applySHlightXYZ(xyz, sh):
  out = applySHlight(xyz, sh)
  # out /= pt.max(out)
  out *= 0.7
  return pt.clip(out, 0, 1)

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

def drawMap(sh, img_size=256):

  n = img_size
  lr = pt.linspace(0, 2 * np.pi, 2 * n)
  ud = pt.linspace(0, np.pi, n)
  ud, lr = pt.meshgrid(ud, lr)

  # we want to make the sphere unwrap at the center of this map,
  # so the left-most column is the furthest-away point on the sphere
  # lr going counter-clockwise = increasing in value.
  # ud starting from 0 (top) to pi (bottom).
  # Lattitude = azimuth = deg from one of xz axis
  # Longtitude = elevation = deg from y-axis
  # In standard unitsphere orientation;
  # z = up (so set y = pt(cos(ud))) ref. https://www.learningaboutelectronics.com/Articles/Spherical-to-cartesian-rectangular-coordinate-converter-calculator.php
  x = -pt.sin(ud) * pt.sin(lr)  # Negative to ensure correct left-right orientation
  y = pt.cos(ud)                # No negative sign needed for up-down orientation
  z = -pt.sin(ud) * pt.cos(lr)  # Negative to ensure correct front-back orientation

  lm = n // 2
  rm = n + (n // 2)

  out = applySHlightXYZ(pt.stack([x, y, z], 0), sh)
  out_centered = out[:, :, lm:rm].clone()
  out_clean = out.clone()
  out[:, :, lm] = pt.tensor((1, 0, 0))[:, None]
  out[:, :, rm] = pt.tensor((1, 0, 0))[:, None]
  return out, out_centered, out_clean
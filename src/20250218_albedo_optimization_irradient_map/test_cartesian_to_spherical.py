
from termcolor import colored
import torch
import numpy as np 
from albedo_optimization import cartesian_to_spherical

def test_cartesian_to_spherical():
    normal = np.array([
        [0,0, 1], # 0,0
        [1,0,0], # [pi/2, 0]
        [0,0,-1], # [pi, 0] or [-pi, 0]
        [-1,0,0], # [-pi/2, 0, ]
        [0,1,0], # [0, pi/2]
        [0,-1,0] # [0, -pi/2]
    ]) # shape [6,3]
    target = np.array([
        [0,0],
        [np.pi/2, 0],
        [np.pi,0],
        [-np.pi/2,0],
        [0, np.pi/2],
        [0, -np.pi/2]
    ])
    target = torch.tensor(target).float()
    normal = torch.tensor(normal).float()
    normal = normal[:,:,None,None]
    assert len(normal.shape) == 4 and normal.shape[0] == 6 and  normal.shape[1] == 3 and normal.shape[2] == 1 and normal.shape[3] == 1 # verify normal shape
    theta_phi = cartesian_to_spherical(normal)
    # devide by pi for easy visualize 
    assert torch.allclose(target, theta_phi[:,:,0,0])
    print(colored('[passed]', 'green'),  'test_cartesian_to_spherical')

if __name__ == "__main__":
    test_cartesian_to_spherical()
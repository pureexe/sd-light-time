import os 
import numpy as np
from sh_utils import get_shcoeff, cartesian_to_spherical, get_ideal_normal_ball_z_up
import skimage

def test_cartesian_to_spherical():
    # testcase 
    xyz = np.zeros((8,3))
    xyz[0] = np.array([1,0,0])
    xyz[1] = np.array([0,1,0])
    xyz[2] = np.array([-1,0,0])
    xyz[3] = np.array([-np.sqrt(2), -np.sqrt(2),0])
    xyz[4] = np.array([0,-1,0])
    xyz[5] = np.array([np.sqrt(2), -np.sqrt(2),0])
    xyz[6] = np.array([0,0,1])
    xyz[7] = np.array([0,0,-1])
    theta, phi = cartesian_to_spherical(xyz)
    
    # test case 1
    expect_theta = np.array([0, 0, 0, 0, 0, 0, np.pi/2, -np.pi/2])
    assert np.allclose(theta, expect_theta)
    expect_phi = np.array([0, np.pi/2, np.pi, np.pi*1.25, np.pi*1.5,  np.pi * 1.75, 0, 0])
    assert np.allclose(phi, expect_phi)
    return True

def test_get_normal_ball():
    normal_map, mask = get_ideal_normal_ball_z_up(1001)
    assert np.allclose(normal_map[0,500], np.array([0,0,1])) # check top pixel
    assert np.allclose(normal_map[1000,500], np.array([0,0,-1])) # check bottom pixel 
    assert np.allclose(normal_map[500,0], np.array([0,-1,0])) # check left pixel     
    assert np.allclose(normal_map[500,1000], np.array([0,1,0])) # check right pixel
    assert np.allclose(normal_map[500,500], np.array([1,0,0])) # check center pixel     
    return True



def main():
    test_get_normal_ball()
    test_cartesian_to_spherical()


    #coeff = get_shcoeff(image, Lmax=100)



if __name__ == "__main__":
    main()
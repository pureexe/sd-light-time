import os
import numpy as np 
import skimage
from shading_integrate import rotate_rays, get_rotation_matrix_from_normal, get_incoming_light, get_incoming_coords, get_ndotl, get_albedo




def test_get_rotation_matrix_from_normal():
    SOURCE_DIRECTION = np.array([0,0,1])[...,None]
    # 6 direction of unit vector
    test_direction = np.array([
        [1,0,0],
        [0,1,0],
        [-1,0,0],
        [0,-1,0],
        [0,0,1],
        [0,0,-1]
    ])[None] # shape [1,6,3]
    
    # get the rotation matrix
    rotated_matrix = get_rotation_matrix_from_normal(test_direction)

    # make sure the rotation matrix get actually rotate back to the same direction
    for i in range(test_direction.shape[1]):
        rotated_normal = np.dot(rotated_matrix[0,i], SOURCE_DIRECTION)[...,0]
        assert np.allclose(rotated_normal, test_direction[0,i])

def test_get_incoming_light():
    """
    Test get incoming light
    """
    envmap = skimage.io.imread("data/coordinates_z_up.png")[...,:3]
    envmap = skimage.img_as_float(envmap)
    rays = np.array([
        [0,0,1], # expect green
        [0,0,-1], # expect yellow
        [0,1,0], # expect cyan
        [0,-1,0], # expect red
        [1,0,0], # expect blue
        [-1,0,0] # expect magenta
    ]) # shape [6,3]
    expected_color = np.array([
        [0, 1, 0], # green
        [1, 1, 0], # yellow
        [0, 1, 1], # cyan
        [1, 0, 0], # red
        [0, 0, 1], # blue
        [1, 0, 1] # magenta
   ])
    rays = rays[None,None] # shape [1,1,6,3]
    coords = get_incoming_coords(rays)
    incoming = get_incoming_light(envmap,rays)

    # check if the incoming light is correct
    assert np.allclose(incoming, expected_color[None,None])

    # save output for eye inspect
    os.makedirs("output/test_shading_integrate", exist_ok=True)
    for i in range(6):
        image = np.ones((100,100,3))
        image = image * incoming[0,0,i]
        image = skimage.img_as_ubyte(image)
        skimage.io.imsave(f"output/test_shading_integrate/incoming_{i}.png", image)

def test_get_ndotl():
    """
    Test get n dot l
    """
    n = np.array([
        [0,0,1], # expect 1
        [1,0,0], # expect 0
        [0,1,0], # expect 0
        [0,0,-1], # expect -1
        [-1,0,0], # expect 0
        [0,-1,0], # expect 0
        [np.sqrt(2)/2, np.sqrt(2)/2, 0], # expect 0
        [0, np.sqrt(2)/2, np.sqrt(2)/2], # expect 0.5
    ]) # shape [8,3]
    n = n[None] # shape [1,8,3]
    l = np.array([
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1]
    ])
    expected_ndotl = np.array([
        [1],
        [0],
        [0],
        [0], # you might expect -1, but we clamp it to 0 because light should not be negative into the surface
        [0],
        [0],
        [0],
        [np.sqrt(2)/2]
    ])
    ndotl = get_ndotl(n,l) 

    # check if the ndotl is correct
    assert np.allclose(ndotl, expected_ndotl[None])
    return True

def test_get_albedo():
    """
    Test get albedo
    """
    H, W = 100, 100
    albedo = get_albedo(H,W)
    assert albedo.shape == (H,W,3)
    # we just expect albedo to be white at the moment
    expected_albedo = np.ones((H,W,3))
    assert np.allclose(albedo, expected_albedo)

def main():
    test_get_rotation_matrix_from_normal()
    test_get_incoming_light()
    test_get_ndotl()f
    test_get_albedo()
    print("All test passed")

if __name__ == "__main__":
    main()
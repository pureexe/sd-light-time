import torch
import numpy as np
from termcolor import colored
import skimage
import os

from albedo_optimization import get_ideal_normal_ball_y_up, n2v, AlbedoOptimization


def test_setup_normal_pipeline():
    pass 

def test_compute_normal():
    pass 

def test_get_normal():
    pass

def test_setup_albedo():
    albedo_optimization = AlbedoOptimization()
    assert albedo_optimization.albedo.shape == (1, 3, 512, 512)
    print(colored('[passed]', 'green'),  'test_setup_albedo') 

def test_setup_spherical_coefficient():
    albedo_optimization = AlbedoOptimization()
    assert albedo_optimization.shcoeffs.shape == (25, 3, 9)
    print(colored('[passed]', 'green'),  'test_setup_spherical_coefficient') 

def test_get_basis():
    normal, mask = get_ideal_normal_ball_y_up(256)
    normal =  torch.tensor(normal).permute(2,0,1)[None]

    albedo_optimization = AlbedoOptimization()
    
    pred_basis = albedo_optimization.get_basis(normal)

    assert pred_basis.shape == (1, 9, 256, 256)

    source_basis = get_basis(normal)
    assert torch.allclose(pred_basis, source_basis)

    print(colored('[passed]', 'green'),  'test_get_basis') 

def test_render_image():
    # we will render image with white 
    # note that. please always eye inspect that everything go as planned 
    OUTPUT_PATH = "output/test_render_image"
    if OUTPUT_PATH != "":
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    # get ball of normal
    normal, mask = get_ideal_normal_ball_y_up(257)
    normal =  torch.tensor(normal).permute(2,0,1)[None]

    albedo_optimization = AlbedoOptimization()
    
    white_albedo = torch.ones(1,3,257,257)

    # render with light is coming from the top
    sh_top = [0., 0., 1., 0., 0., 0., 0., 0., 0.]
    sh_top = torch.tensor([sh_top])
    sh_top = torch.cat([sh_top[:,None,:],sh_top[:,None,:],sh_top[:,None,:]], axis=1)
    assert sh_top.shape == (1,3,9)
 
    image_top = albedo_optimization.render_image(sh_top, normal, white_albedo)
    image_top = n2v(image_top) 
    # make sure top color is correct
    assert torch.allclose(image_top[0,:,0,128].float(), torch.tensor([1., 1., 1.]).float())

    image_top = image_top[0].permute(1,2,0).numpy()
    if OUTPUT_PATH != "":
        skimage.io.imsave(os.path.join(OUTPUT_PATH, "sh_top.png"), skimage.img_as_ubyte(image_top))

    # render with light is coming from the bottom (red channel only)
    sh_bottom = [0., 0., -1., 0., 0., 0., 0., 0., 0.]
    sh_bottom = torch.tensor([sh_bottom])
    sh_bottom = torch.cat([
        sh_bottom[None],
        torch.zeros(1,1,9),
        torch.zeros(1,1,9)
    ], axis=1)
    assert sh_bottom.shape == (1,3,9)
    image_bottom = albedo_optimization.render_image(sh_bottom, normal, white_albedo)
    image_bottom = n2v(image_bottom)
    assert torch.allclose(image_bottom[0,:,256,128].float(), torch.tensor([1., .5, .5]).float())
    
    image_bottom = image_bottom[0].permute(1,2,0).numpy()
    if OUTPUT_PATH != "":
        skimage.io.imsave(os.path.join(OUTPUT_PATH, "sh_bottom.png"), skimage.img_as_ubyte(image_bottom))

    # render with light is coming from the left (green channel only)
    sh_left = [0., -1., 0., 0., 0., 0., 0., 0., 0.]
    sh_left = torch.tensor([sh_left])
    sh_left = torch.cat([
        torch.zeros(1,1,9),
        sh_left[None],
        torch.zeros(1,1,9)
    ], axis=1)
    assert sh_left.shape == (1,3,9)
    image_left = albedo_optimization.render_image(sh_left, normal, white_albedo)
    image_left = n2v(image_left)
    assert torch.allclose(image_left[0,:,128,0].float(), torch.tensor([.5, 1., .5]).float())
    
    image_left = image_left[0].permute(1,2,0).numpy()
    if OUTPUT_PATH != "":
        skimage.io.imsave(os.path.join(OUTPUT_PATH, "sh_left.png"), skimage.img_as_ubyte(image_left))

    # render with light is coming from the right (blue channel only)
    sh_right = [0., 1., 0., 0., 0., 0., 0., 0., 0.]
    sh_right = torch.tensor([sh_right])
    sh_right = torch.cat([
        torch.zeros(1,1,9),
        torch.zeros(1,1,9),
        sh_right[None]
    ], axis=1)
    assert sh_right.shape == (1,3,9)
    image_right = albedo_optimization.render_image(sh_right, normal, white_albedo)
    image_right = n2v(image_right)
    assert torch.allclose(image_right[0,:,128,256].float(), torch.tensor([.5, .5, 1.]).float())
    
    image_right = image_right[0].permute(1,2,0).numpy()
    if OUTPUT_PATH != "":
        skimage.io.imsave(os.path.join(OUTPUT_PATH, "sh_right.png"), skimage.img_as_ubyte(image_right))

    print(colored('[passed]', 'green'),  'test_render_image') 


def test_on_train_epoch_start():
    pass

def test_training_step():
    pass

def test_log_tensorboard():
    pass

def test_validation_step():
    pass

def test_get_ideal_normal_ball_y_up():
    """
    get_ideal_normal_ball_y_up is a function that returns the ideal normal map for a sphere with the y-axis pointing up.
    """
    normal, mask = get_ideal_normal_ball_y_up(257)
    assert normal.shape == (257, 257, 3) # shape should be match
    assert np.allclose(normal[128,0], [-1,0,0]) # left is x-minus 
    assert np.allclose(normal[128,256], [1,0,0]) # right is x-plus
    assert np.allclose(normal[0,128], [0,1,0]) # top is y-plus
    assert np.allclose(normal[256,128], [0,-1,0]) # bottom is y-minus
    assert np.allclose(normal[128,128], [0,0,1]) # middle is z-plus
    
    print(colored('[passed]', 'green'),  'test_get_ideal_normal_ball_y_up') 

def test_n2v():
    """
    n2v is design for make tnesor in range [-1,1] to [0,1] for visualize
    """
    ones = torch.ones(1,3,3, 3)
    zeros = torch.zeros(1,3,3,3)
    minus_ones = -1 * torch.ones(1,3,3,3)
    assert torch.all(torch.isclose(n2v(ones),torch.ones(1,3,3,3)))
    assert torch.all(torch.isclose(n2v(zeros),torch.ones(1,3,3,3) * 0.5))
    assert torch.all(torch.isclose(n2v(minus_ones),torch.zeros(1,3,3,3)))

    #TODO: Implement test for n2v in lab color space
    print(colored('[passed]', 'green'),  'test_n2v')

### RESOURCE FOR TEST FUNCTION ###
def get_basis(normal_images):
    # THIS IS THE LESS SLIGHTLY MODIFIED VERSION OF THE FUNCTION to make sure output is the same
    """
    @see https://github.com/diffusion-face-relighting/difareli_code/blob/2dd24a024f26d659767df6ecc8da4ba47c55e7a8/guided_diffusion/models/renderer.py#L25
    Apply the SH to normals(features map)
    :param h: normals(features map) in [B x #C_Normals x H xW]
    """
    th = torch
    pi = np.pi
    num_SH = 9

    constant_factor = th.tensor([1/np.sqrt(4*pi), 
                                        ((2*pi)/3)*(np.sqrt(3/(4*pi))), 
                                        ((2*pi)/3)*(np.sqrt(3/(4*pi))),
                                        ((2*pi)/3)*(np.sqrt(3/(4*pi))), 
                                        (pi/4)*(3)*(np.sqrt(5/(12*pi))), 
                                        (pi/4)*(3)*(np.sqrt(5/(12*pi))),
                                        (pi/4)*(3)*(np.sqrt(5/(12*pi))), 
                                        (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), 
                                        (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
    N = normal_images
    sh = th.stack([
            N[:,0]*0.+1.,   # 1
            N[:,0],         # X
            N[:,1],         # Y
            N[:,2],         # Z
            N[:,0]*N[:,1],  # X*Y
            N[:,0]*N[:,2],  # X*Z
            N[:,1]*N[:,2],  # Y*Z
            N[:,0]**2 - N[:,1]**2,  # X**2 - Y**2
            3*(N[:,2]**2) - 1,      # 3(Z**2) - 1
            ], 
            1) # [bz, 9, h, w]
    
    sh = sh[:, :num_SH, :, :]
    sh = sh.type_as(N) * constant_factor[None, :, None, None].type_as(N)
    return sh


def main():
    print("Running tests for AlbedoOptimization")
    #test_n2v()
    #test_get_ideal_normal_ball_y_up()
    #test_get_basis()
    #test_render_image()
    test_setup_albedo()
    test_setup_spherical_coefficient()
    print(colored('=== ALL TEST PASSED ===', 'green')) 
if __name__ == "__main__":
    main()
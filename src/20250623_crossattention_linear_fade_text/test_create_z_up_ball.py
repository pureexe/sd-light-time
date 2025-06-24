import numpy as np
import skimage

def get_ideal_normal_ball_z_up(size):
    """
    Generate front size of normal ball that has z up
    """
    y = np.linspace(-1, 1, size)
    z = np.linspace(1, -1, size)
    y, z = np.meshgrid(y, z)
    
    # avoid negative value
    x2 = 1 - y**2 - z**2
    mask = x2 >= 0

    # get real x value
    x = np.sqrt(np.clip(x2,0,1))    

    x = x * mask
    y = y * mask
    z = z * mask
    # set x outside mask to be 1
    x = x + (1 - mask)
    normal_map = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

    return normal_map, mask

ball, mask = get_ideal_normal_ball_z_up(256)
ball = (ball + 1.0) / 2.0
ball = skimage.img_as_ubyte(ball)
skimage.io.imsave('normal_ball_z_up.png', ball)
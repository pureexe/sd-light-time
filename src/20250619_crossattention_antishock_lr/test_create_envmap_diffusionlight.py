from vll_datasets.diffusionrenderer_mapper import envmap_vec
import skimage 
import numpy as np

def get_light_dir( height = 256, width = 512):
    # create  from [0, 2pi] and [-pi/2, pi/2] 
    #theta = np.linspace(0, 2 * np.pi, width)
    theta = np.linspace(0,  2 * np.pi, width)
    phi = np.linspace(np.pi / 2, -np.pi / 2, height)
    theta, phi = np.meshgrid(theta, phi)
    x = np.cos(theta) * np.cos(phi)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(phi)
    light_dir = np.stack([x, y, z], axis=-1)  # Shape: (height, width, 3)
    return light_dir.astype(np.float32)  # Convert to float32 for consistency


# def from_pyshtool_to_diffusionrenderer(light_dir):
#     """
#     Convert light direction from x-to_monitor,y-left,z-up to x-right,y-up,z-to_scene convention.
#     input: light_dir (np.array) - [..., 3] in x-to_monitor,y-left,z-up convention
#     output: light_dir (np.array) - [..., 3] in x-right,y-up,z-to_scene convention
#     """
#     transformation_matrix = np.array([[0, 1, 0],
#                                        [0, 0, 1],
#                                        [-1, 0, 0]], dtype=np.float32)
#     original_shape = light_dir.shape[:-1]
#     light_dir = light_dir.reshape(-1, 3)  # Flatten the last dimension
#     light_dir = np.dot(light_dir, transformation_matrix.T)
#     return light_dir.reshape(*original_shape, 3)  # Reshape back to original shape with 3 channels

def from_pyshtool_to_diffusionrenderer(light_dir):
    """
    Convert light direction from x-to_monitor,y-left,z-up to x-right,y-up,z-to_scene convention.
    input: light_dir (np.array) - [..., 3] in x-to_monitor,y-left,z-up convention
    output: light_dir (np.array) - [..., 3] in x-right,y-up,z-to_scene convention
    """
    transformation_matrix = np.array([[0, -1, 0],
                                       [0, 0, 1],
                                       [-1, 0, 0]], dtype=np.float32)
    original_shape = light_dir.shape[:-1]
    light_dir = light_dir.reshape(-1, 3)  # Flatten the last dimension
    light_dir = np.dot(light_dir, transformation_matrix.T)
    return light_dir.reshape(*original_shape, 3)  # Reshape back to original shape with 3 channels




def flip_inside_outside(light_dir):
    """
    Flip the inside-outside of the light direction.
    input: light_dir (np.array) - [..., 3] in x-right,y-up,z-to_scene convention
    output: light_dir (np.array) - [..., 3] in x-right,y-up,z-to_scene convention with inside-outside flipped
    """
    return light_dir * -1


def main():
    envmap = get_light_dir()
    envmap = from_pyshtool_to_diffusionrenderer(envmap)
    # envmap = flip_inside_outside(envmap)
    print(envmap.shape)  # Should print: torch.Size([256, 512, 3])
    envmap = (envmap + 1.0) / 2.0
    envmap = skimage.img_as_ubyte(envmap)
    skimage.io.imsave('normal_envmap_diffusionlight_converted.png', envmap)

if __name__ == "__main__":
    main()
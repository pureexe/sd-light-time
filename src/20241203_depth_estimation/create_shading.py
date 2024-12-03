import torch 
import numpy as np
from PIL import Image
import skimage

def add_SHlight(self, normal_images, sh_coeff):
    '''
        sh_coeff: [bz, 9, 3]
    '''
    N = normal_images
    sh = torch.stack([
            N[:,0]*0.+1., N[:,0], N[:,1], \
            N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
            N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
            ], 
            1) # [bz, 9, h, w]
    sh = sh*self.constant_factor[None,:,None,None]
    shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
    return shading



def normalize_last_channel(arr):
    """
    Normalizes the last channel of a numpy array (H, W, 3) to ensure it becomes a unit vector.
    
    Parameters:
    arr (numpy.ndarray): Input array of shape (H, W, 3).
    
    Returns:
    numpy.ndarray: The normalized array where the last channel is a unit vector for each pixel.
    """
    if arr.shape[-1] != 3:
        raise ValueError("Input array must have shape (H, W, 3).")
    
    # Compute the magnitude (Euclidean norm) of the last channel
    magnitude = np.linalg.norm(arr, axis=-1, keepdims=True)
    
    # Avoid division by zero by setting zero magnitudes to 1 (effectively skipping normalization for zero vectors)
    magnitude = np.where(magnitude == 0, 1, magnitude)
    
    # Normalize by dividing the array by the magnitude
    normalized_arr = arr / magnitude
    return normalized_arr

def main():
    normal = np.load("normal_bae.npy")
    normal = normalize_last_channel(normal)

    # load lighting 
    

    print(normal.min())
    exit()

if __name__ == "__main__":
    main()
import torch
import skimage 
import numpy as np 

def sobel_edge_detection(tensor):
    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    # Move filters to the same device as tensor
    device = tensor.device
    sobel_x, sobel_y = sobel_x.to(device), sobel_y.to(device)
    
    # Convert to grayscale
    tensor_gray = 0.2989 * tensor[:, 0:1, :, :] + 0.5870 * tensor[:, 1:2, :, :] + 0.1140 * tensor[:, 2:3, :, :]

    # Apply Sobel filters
    edge_x = torch.nn.functional.conv2d(tensor_gray, sobel_x, padding=1)
    edge_y = torch.nn.functional.conv2d(tensor_gray, sobel_y, padding=1)
    
    # Compute edge magnitude
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    
    return edge        


image = skimage.io.imread("/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/test/control_render_from_fitting_v2/everett_dining1/dir_0_mip2.png")
image = skimage.img_as_float(image)
image = torch.tensor(image).float()
image = image.permute(2,0,1)[None]
image = sobel_edge_detection(image)
image = image[0].numpy()
print("SOBEL MAX:", image.max())
print("SOBEL MIN:",image.min())
image = np.clip(image, 0, 1)
image = skimage.img_as_ubyte(image)
skimage.io.imsave("sobel_edge.png", image)



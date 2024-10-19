import os
import re
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils

# Define the folder where the images are stored
image_folder = 'src/20241018/output'
output_grid_path = 'src/20241018/output_image_grid.png'

# Define the regex pattern to match the filenames
pattern = r'image_p([01])_timestep([0-9]+)\.png'

# Initialize dictionaries to store images based on 'p' value
images_p0 = []
images_p1 = []

# Define image transformations
transform = T.Compose([
    T.Resize((256, 256)),  # Resize images to a fixed size (optional)
    T.ToTensor(),          # Convert image to tensor
])

# Load images from the folder
for filename in sorted(os.listdir(image_folder)):
    match = re.match(pattern, filename)
    if match:
        p_value, timestep = match.groups()
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        
        if p_value == '0':
            images_p0.append(image_tensor)
        else:
            images_p1.append(image_tensor)

# Stack images into grid format
if images_p0 and images_p1:
    images_p0_grid = vutils.make_grid(torch.stack(images_p0), nrow=10, padding=2)
    images_p1_grid = vutils.make_grid(torch.stack(images_p1), nrow=10, padding=2)
    
    # Combine both grids vertically
    combined_grid = torch.cat((images_p0_grid, images_p1_grid), dim=1)

    # Save or display the image grid
    vutils.save_image(combined_grid, output_grid_path)
    print(f"Image grid saved to {output_grid_path}")
else:
    print("No images found or unable to create grids.")
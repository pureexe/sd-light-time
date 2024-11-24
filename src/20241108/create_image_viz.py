import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import json

def create_image_grid(input_dir, output_path, dataset_val_dir, columns=3, image_size=(256, 256)):
    """
    Reads images from a directory, creates an image grid, and saves it to a file.
    
    Args:
        input_dir (str): Path to the directory containing images.
        output_path (str): Path to save the generated image grid.
        columns (int): Number of columns in the grid. Default is 10.
        image_size (tuple): Resize all images to this size. Default is (128, 128).
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        raise ValueError(f"The directory {input_dir} does not exist.")
    
    # Read and sort image file names
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    if not image_files:
        raise ValueError(f"No images found in the directory {input_dir}.")
    
    #with open(index_path) as f:
    #    index_file = json.load(f)
 
    # Load images
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    images = []
    
    for ident_id in range(5):
        # first image is blank image 
        images.append(torch.zeros((3, ) + image_size).float())
        # add shading mask 
        for light_id in range(3):
            img_path = dataset_val_dir+'/shadings/00000/'+f"{ident_id * 3 + light_id:05d}"+".png"
            images.append(transform(Image.open(img_path).convert('RGB')))
        # add input image 
        img_path = dataset_val_dir+'/images/00000/'+f"{ident_id * 3:05d}"+".jpg"
        images.append(transform(Image.open(img_path).convert('RGB')))
        # show light
        for light_id in range(3):
            images.append(transform(Image.open(os.path.join(input_dir, image_files[ident_id * 3 + light_id])).convert('RGB')))




    #dataset_dir = os.path.dirname(index_path)

    # for image_name in index_file['envmap_index'][0]:
    #     img_path = dataset_dir+'/shadings/'+image_name+".png"
    #     images.append(transform(Image.open(img_path).convert('RGB')))

    # for image_id in range(columns):
    #     img_path = dataset_val_dir+'/shadings/00000/'+f"{image_id:05d}"+".png"
    #     print(img_path)
    #     images.append(transform(Image.open(img_path).convert('RGB')))

    # for idx, img_file in enumerate(image_files): 
    #     if idx % 10 == 0:
    #         image_name = index_file['image_index'][idx // 10]
    #         img_path = dataset_dir+'/images/'+image_name+".jpg"
    #         images.append(transform(Image.open(img_path).convert('RGB')))
    #     images.append(transform(Image.open(os.path.join(input_dir, img_file)).convert('RGB')))
    #images = [transform(Image.open(os.path.join(input_dir, img_file)).convert('RGB')) for img_file in image_files]

    # Create the grid
    grid = make_grid(images, nrow=columns+1, padding=2)

    # Save the grid image using torchvision
    save_image(grid, output_path)
    print(f"Image grid saved to {output_path}")


datasets = ['viz']
dataset_dirs = [
    '/ist/ist-share/vision/relight/datasets/face/ffhq_defareli/viz',
]



image_sources = [
    'output/20241108/val_{}/default/1.0/adagan_only/1e-4/chk11/lightning_logs',
    'output/20241108/val_{}/default/1.0/adagan_only/1e-5/chk10/lightning_logs',
    'output/20241108/val_{}/default/1.0/controlnet_only/1e-4/chk8/lightning_logs',
    'output/20241108/val_{}/default/1.0/controlnet_only/1e-5/chk8/lightning_logs',
    'output/20241108/val_{}/default/1.0/mint_pretrain/1e-4/chk42/lightning_logs',
    'output/20241108/val_{}/default/1.0/mint_pretrain/1e-5/chk43/lightning_logs',
    'output/20241108/val_{}/default/1.0/mint_scrath/1e-5/chk34/lightning_logs',
    'output/20241108/val_{}/default/1.0/mint_scrath/5e-6/chk34/lightning_logs'
]
exp_names = [
    'output/20241108/grid_viz/{}_adagan_only_1e-4.jpg',
    'output/20241108/grid_viz/{}_adagan_only_1e-5.jpg',
    'output/20241108/grid_viz/{}_controlnet_only_1e-4.jpg',
    'output/20241108/grid_viz/{}_controlnet_only_1e-5.jpg',
    'output/20241108/grid_viz/{}_mint_pretrain_1e-4.jpg',
    'output/20241108/grid_viz/{}_mint_pretrain_1e-5.jpg',
    'output/20241108/grid_viz/{}_mint_scrath_1e-5.jpg',
    'output/20241108/grid_viz/{}_mint_scrath_5e-6.jpg',
]

# ../../output/20241108/val_train2right/default/1.0/adagan_only/1e-4/chk11/lightning_logs/version_90711/crop_image
# ../../output/20241108/val_train2right/default/1.0/adagan_only/1e-5/chk10/lightning_logs/version_90712/crop_image
# ../../output/20241108/val_train2right/default/1.0/controlnet_only/1e-4/chk8/lightning_logs/version_90709/crop_image
# ../../output/20241108/val_train2right/default/1.0/controlnet_only/1e-5/chk8/lightning_logs/version_90710/crop_image
# ../../output/20241108/val_train2right/default/1.0/mint_pretrain/1e-4/chk42/lightning_logs/version_90626/crop_image
# ../../output/20241108/val_train2right/default/1.0/mint_pretrain/1e-5/chk43/lightning_logs/version_90706/crop_image
# ../../output/20241108/val_train2right/default/1.0/mint_scrath/5e-6/chk34/lightning_logs/version_90708/crop_image

for dataset_id, dataset in enumerate(datasets):
    for image_id in range(len(image_sources)):
        input_dir = image_sources[image_id].format(dataset)
        version_dir = os.listdir(input_dir)[0]
        input_dir = os.path.join(input_dir, version_dir, "crop_image")
        output_file = exp_names[image_id].format(dataset)
        create_image_grid(input_dir, output_file, dataset_dirs[dataset_id])
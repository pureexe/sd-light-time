
ROOT_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20241108/val_valid_face_different/default/1.0/v2a_adagn_only_shcoeff/1e-5/chk29/lightning_logs/version_93680/"
PREDICT_DIR = ROOT_DIR + "/crop_image"
SCORE_DIR = ROOT_DIR + "/calculated_score"
GROUNDTRUTH_DIR = "/ist/ist-share/vision/relight/datasets/face/ffhq_defareli/diffusion-face-relight-testset-different-subject/images"
INDEX_FILE = "/ist/ist-share/vision/relight/datasets/face/ffhq_defareli/diffusion-face-relight-testset-different-subject/multipie_testset2.json"
SEGMENT_DIR = "/ist/ist-share/vision/relight/datasets/face/ffhq_defareli/diffusion-face-relight-testset-different-subject/segments"

import pandas as pd
import numpy as np
import torch
import json
import lpips
import os 
import torchvision
from PIL import Image
from tqdm.auto import tqdm
import torchmetrics
import skimage

def load_and_preprocess_image(image_path):
    """
    Loads an image from the given path, normalizes its pixel values to [0, 1],
    and resizes it to [3, 256, 256].

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape [3, 256, 256].
    """
    # Define the transformation pipeline
    transform_pipeline = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),       # Resize to 128x128
        torchvision.transforms.ToTensor(),              # Convert to tensor and normalize to [0, 1]
    ])

    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Apply the transformations
    image_tensor = transform_pipeline(image)

    # Assert to ensure image is in range [0, 1]
    assert image_tensor.min() >= 0.0 and image_tensor.max() <= 1.0, "Image tensor values are not in range [0, 1]"

    return image_tensor

def get_face_mask(face_name):
    mask_path = os.path.join(SEGMENT_DIR,f'anno_{face_name}.png')
    face_segment_anno =skimage.io.imread(mask_path)
    skin = (face_segment_anno == 1)
    l_brow = (face_segment_anno == 2)
    r_brow = (face_segment_anno == 3)
    l_eye = (face_segment_anno == 4)
    r_eye = (face_segment_anno == 5)
    eye_g = (face_segment_anno == 6)
    l_ear = (face_segment_anno == 7)
    r_ear = (face_segment_anno == 8)
    ear_r = (face_segment_anno == 9)
    nose = (face_segment_anno == 10)
    mouth = (face_segment_anno == 11)
    u_lip = (face_segment_anno == 12)
    l_lip = (face_segment_anno == 13)
    face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip))
    return face



def calculate_score(pair, lpips_loss, ssim_loss):
    face_name = pair['src'].replace(".png","")
    pred_name = pair['src'].replace(".png","") + "_" + pair['dst'].replace(".png","")
    gt_name = pair['gt'].replace(".png","")

    # load image 
    pred_image = load_and_preprocess_image(os.path.join(PREDICT_DIR, pred_name + ".png"))
    gt_image = load_and_preprocess_image(os.path.join(GROUNDTRUTH_DIR, gt_name + ".png"))

    # load mask
    face_mask = get_face_mask(face_name)
    face_mask = torch.from_numpy(face_mask)[None]
    face_mask = torchvision.transforms.functional.resize(face_mask, [256,256])

    #apply mask to both pred and gt 
    pred_image = pred_image * face_mask
    gt_image = gt_image * face_mask

    # change shape to BCHW
    pred_image = pred_image[None]
    gt_image = gt_image[None]

    # compute SSIM
    ssim = ssim_loss(gt_image, pred_image)
    ddsim = (1.0 - ssim) / 2.0

    # compute lpips
    normalize_pt_image = pred_image * 2.0 - 1.0
    normalize_gt_image = gt_image * 2.0 - 1.0
    lpips = lpips_loss(normalize_pt_image, normalize_gt_image)

    # compute MSE 
    mse = torch.nn.functional.mse_loss(gt_image, pred_image, reduction="none").mean()
    psnr = -10 * torch.log10(mse)
    return {
        'mse': mse.item(),
        'ddsim': ddsim.item(),
        'lpips': lpips.item(),
        'ssim': ssim.item(),
        'psnr': psnr.item()
    }


def save_score(pair, score):
    pred_name = pair['src'].replace(".png","") + "_" + pair['dst'].replace(".png","")
    os.makedirs(SCORE_DIR, exist_ok=True)
    for key in ['ddsim', 'mse', 'lpips', 'ssim', 'psnr']:
        os.makedirs(SCORE_DIR + "/"+key, exist_ok=True)
        with open(f"{SCORE_DIR}/{key}/{pred_name}.txt", "w") as f:
            f.write(f"{score[key]}\n")

@torch.inference_mode()
def main():
    # read index
    lpips_loss = lpips.LPIPS(net='alex')
    ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)  

    data = {
        "ddsim": [],
        "mse": [],
        "lpips": [],
        "ssim": [],
        "psnr": []
    }

    with open(INDEX_FILE) as f:
        info = json.load(f)

    for k in tqdm(info['pair'].keys()):
        pair = info['pair'][k]
        score = calculate_score(pair, lpips_loss, ssim_loss)
        save_score(pair, score)
        for key in ['ddsim', 'mse', 'lpips', 'ssim', 'psnr']:
            data[key].append(score[key])

    

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Calculate the mean of psnr and ssim
    mean_values = df.mean().to_frame(name="Mean").T

    # Display the mean as a table
    print(mean_values)

    mean_values.to_csv(SCORE_DIR + "/mean_values.csv", index=True)


if __name__ == "__main__":
    main()
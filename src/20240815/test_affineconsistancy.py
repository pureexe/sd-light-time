
import pytest
import torch
from AffineConsistancy import AffineConsistancy
from LightEmbedingBlock import set_light_direction
from PIL import Image
import numpy as np




# 0.  Make sure that model is created correctly
@pytest.mark.parametrize("learning_rate,use_consistancy_loss", [("1e-4", False), ("1e-4", True)])
def test_load_affine_consistancy(learning_rate, use_consistancy_loss):
    learning_rate = 1e-4
    model = AffineConsistancy(learning_rate=learning_rate, use_consistancy_loss = use_consistancy_loss)

# 1. Make sure that model is use SD1.5 as fp16 mode 
def test_sd_version():
    learning_rate, use_consistancy_loss = 1e-4, False
    prompt = "a photo of an astronaut riding a horse on mars"
    target_image = "cat_image.png"

    model = AffineConsistancy(learning_rate=learning_rate, use_consistancy_loss = use_consistancy_loss)
    # check if it 16bit
    assert model.pipe.unet.dtype == torch.float16
    # disable light block 
    set_light_direction(model.pipe.unet, None, is_apply_cfg=True)
    image = model.pipe(
        prompt,
        generator=torch.Generator(device=model.device).manual_seed(42),
        guidance_scale=7.5,
        num_inference_steps=50,
    ).images[0]  

    assert image.size == (512, 512)

    # read the image 
    target_image = Image.open("assets/test_case/sd1.5_fp16/astronaut_rides_horse.png")
    target_image = np.array(target_image) / 255.0 

    # check if the image is match
    image = np.array(image) / 255.0
    #assert np.allclose(image, target_image, atol=1e-3)
    assert np.abs(image - target_image).mean() < 0.01


@pytest.mark.parametrize("guidance_scale", [1.0, 3.0, 5.0, 7.0])
def test_set_guidance_scale(guidance_scale):
    learning_rate = 1e-4
    use_consistancy_loss = False
    model = AffineConsistancy(learning_rate=learning_rate, use_consistancy_loss = use_consistancy_loss)
    model.set_guidance_scale(guidance_scale)
    assert model.guidance_scale == guidance_scale


def test_get_vae_features():
    pass 

def test_get_light_features():
    pass

def test_compute_train_loss():
    pass

def test_light_add_block():
    # check if the network actually add light block to the unet
    pass

def test_set_light_direction():
    # check if the light direction is set correctly
    pass


# 2. test that we load correct depth controlnet 
# 6. test that circle mask is create correctly


# 4. test that light block shape is match the vae 

# 5. test that unet is register as trainable parameter 

# 7. test if the vae ifeature is compute corerectly 

# 8 the that get_light_feautres function is work correctly 

# 9. test that get_envmap_consisitancy is work correctly


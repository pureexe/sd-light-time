cd /ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250519_epoch_resample

python inference.py -p 1e-4_lstsq_image_lstsq_shading -m rotate_everett_kitchen6_least_square_shading,rotate_everett_dining1_least_square_shading,rotate_everett_kitchen2_least_square_shading,rotate_everett_kitchen4_least_square_shading

python inference.py -p 1e-5_lstsq_image_lstsq_shading -m rotate_everett_kitchen6_least_square_shading,rotate_everett_dining1_least_square_shading,rotate_everett_kitchen2_least_square_shading,rotate_everett_kitchen4_least_square_shading

python inference.py -p 1e-4_real_image_lstsq_shading -m rotate_everett_kitchen6_least_square_shading,rotate_everett_dining1_least_square_shading,rotate_everett_kitchen2_least_square_shading,rotate_everett_kitchen4_least_square_shading

python inference.py -p 1e-5_real_image_lstsq_shading -m rotate_everett_kitchen6_least_square_shading,rotate_everett_dining1_least_square_shading,rotate_everett_kitchen2_least_square_shading,rotate_everett_kitchen4_least_square_shading

python inference.py -p 1e-4_real_image_lstsq_shading -m everett_kitchen6_diffusionlight_shading,everett_dining1_diffusionlight_shading,everett_kitchen2_diffusionlight_shading,everett_kitchen4_diffusionlight_shading

python inference.py -p 1e-5_real_image_lstsq_shading -m everett_kitchen6_diffusionlight_shading,everett_dining1_diffusionlight_shading,everett_kitchen2_diffusionlight_shading,everett_kitchen4_diffusionlight_shading

python inference.py -p 1e-4_real_image_lstsq_shading -m everett_kitchen6_diffusionlight_shading,everett_dining1_diffusionlight_shading,everett_kitchen2_diffusionlight_shading,everett_kitchen4_diffusionlight_shading

python inference.py -p 1e-5_real_image_lstsq_shading -m everett_kitchen6_diffusionlight_shading,everett_dining1_diffusionlight_shading,everett_kitchen2_diffusionlight_shading,everett_kitchen4_diffusionlight_shading
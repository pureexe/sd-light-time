print("import component")
from chrislib.data_util import load_image

from intrinsic.pipeline import load_models, run_pipeline

# load the models from the given paths
print("loading model...")
models = load_models('v2')

print("loading image...")
# load an image (np float array in [0-1])
image = load_image('/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom1/dir_0_mip2.jpg')

# run the model on the image using R_0 resizing
print("predicting...")
results = run_pipeline(models, image)

print("look data...")
albedo = results['hr_alb']
diffuse_shading = results['dif_shd']
residual = results['residual']
# + multiple other keys for different intermediate components

print("albedo")
print(albedo.shape)
print(albedo.min())
print(albedo.max())

print("diffuse_shading")
print(diffuse_shading.shape)
print(diffuse_shading.min())
print(diffuse_shading.max())

print("residual")
print(residual.shape)
print(residual.min())
print(residual.max())
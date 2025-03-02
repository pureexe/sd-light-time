import ezexr
import numpy as np
def main():
    # img = ezexr.imread("/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/control_shading_from_fitting_v3_exr/14n_copyroom10/dir_0_mip2.exr")
    # print(img.shape)
    # print(img.min())
    # print(img.max())
    img = np.ones((512,512,3))
    ezexr.imsave("one_shading.exr", img)


if __name__ == "__main__":
    main()
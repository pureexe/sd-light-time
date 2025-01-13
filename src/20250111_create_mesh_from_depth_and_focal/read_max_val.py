import ezexr 

def main():

    image = ezexr.imread("/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250111_create_mesh_from_depth_and_focal/background_hdr_order100.exr")
    print(image.max())
    exit()
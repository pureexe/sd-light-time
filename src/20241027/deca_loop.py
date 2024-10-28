import os 

PATTERN = "../../output/20241027/{}/no_control/{}/no_control/0.0001/chk{}/lightning_logs/version_87762"

A = "val_coeff27_faceval10k_fuse_test_right"
B = "1.0"
C = "0"

MODES = ["val_coeff27_faceval10k_fuse_test_right", "val_coeff27_faceval10k_fuse_test_left"]
GUIDANCES = ["1.0","3.0","5.0","7.0"]
CHECKPOINTS = ["0","10","20"]

for mode in MODES:
    for guidance in GUIDANCES:
        for ckpt in CHECKPOINTS:
            main_dir = PATTERN.format(mode,guidance,ckpt) 
            if not os.path.exists(main_dir):
                continue
            input_dir = main_dir + "/crop_image"
            output_dir = main_dir + "/face_light"
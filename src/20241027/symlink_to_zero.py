import os 

PATTERN = "../../output/20241027/{}/no_control/{}/no_control/0.0001/chk{}/lightning_logs/"
PATTERN_WEB = "../../output/20241027_web/right/g{}_chk{}"


A = "val_coeff27_faceval10k_fuse_test_left"
B = "1.0"
C = "0"

#MODES = ["val_coeff27_faceval10k_fuse_test_right", "val_coeff27_faceval10k_fuse_test_left"]
MODES = ["val_coeff27_faceval10k_fuse_test_right"]
#MODES = ["val_coeff3_faceval10k_fuse_test_right", "val_coeff3_faceval10k_fuse_test_left"]
GUIDANCES = ["1.0","3.0","5.0","7.0"]
CHECKPOINTS = ["0","10","20"]

for mode in MODES:
    for idx, guidance in enumerate(GUIDANCES):
        for ckpt in CHECKPOINTS:
            main_dir = PATTERN.format(mode,guidance,ckpt) 
            web_dir = PATTERN_WEB.format(guidance.replace(".0",""),ckpt) 
            if not os.path.exists(main_dir):
                print("NOT FOUND: ", main_dir)
                continue
            version_dirs = [c for c in sorted(os.listdir(main_dir)) if c.startswith('version_')]
            version_dir = version_dirs[-1]
            cmd = f'ln -s {main_dir}{version_dir} {web_dir}'
            #print(cmd)
            os.system(cmd)

            #input_dir = main_dir + '/' + version_dir + "/crop_image"
            #output_dir = main_dir + '/' + version_dir + "/face_light"
            #if os.path.exists(output_dir):
            #    print("FOUND: ", output_dir)
            #    continue
            #os.system(f"python demos/demo_reconstruct.py -i {input_dir} -s {output_dir} --saveLight true")
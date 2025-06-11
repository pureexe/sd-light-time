LORA RANK = 4,16,64,256
LR = 1e-4, 1e-5

cd /pure/t1/project/sd-light-time/src/20250609_diffusionrender







#####################
#RUN 01

# version 7259
python train.py -lr 1e-4 --lora_rank 4 

# version 7256
python train.py -lr 1e-4 --lora_rank 16

# version 7257
python train.py -lr 1e-4 --lora_rank 64

# version 117153
python train.py -lr 1e-4 --lora_rank 256

# version 117154
python train.py -lr 1e-5 --lora_rank 4

# version 117161
python train.py -lr 1e-5 --lora_rank 16

# version 117159
python train.py -lr 1e-5 --lora_rank 64

# version  117160
python train.py -lr 1e-5 --lora_rank 256


#############################
# RUN 02 
# version 7272
cd /pure/t1/project/sd-light-time/src/20250609_diffusionrender

python train.py -lr 1e-4 --lora_rank 4 

# version 7273
python train.py -lr 1e-4 --lora_rank 16

# version 7274
python train.py -lr 1e-4 --lora_rank 64

# version 117169
python train.py -lr 1e-4 --lora_rank 256

# version 117170
python train.py -lr 1e-5 --lora_rank 4

# version 117171
python train.py -lr 1e-5 --lora_rank 16

# version 117172
python train.py -lr 1e-5 --lora_rank 64

# version  117173
python train.py -lr 1e-5 --lora_rank 256
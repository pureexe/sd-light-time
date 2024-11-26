# run: LR1e-4 rank 32

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-4_rank32" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=32 \
  --seed="0"


# run: LR1e-4 rank 4

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-4_rank4" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=4 \
  --mixed_precision=fp16 \
  --seed="0"

# run: LR1e-4 rank 8

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-4_rank8" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=8 \
  --mixed_precision=fp16 \
  --seed="0"

# run: LR1e-4 rank 16
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-4_rank16" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=16 \
  --mixed_precision=fp16 \
  --seed="0"
# run: LR1e-4 rank 32
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-4_rank32" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=32 \
  --mixed_precision=fp16 \
  --seed="0"


#########################################

# run: LR1e-5/ rank 4

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-5_rank4" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=4 \
  --mixed_precision=fp16 \
  --seed="0"


# run: LR1e-5/ rank 8
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-5_rank8" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=8 \
  --mixed_precision=fp16 \
  --seed="0"

# run: LR1e-5/ rank 16
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-5_rank16" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=16 \
  --mixed_precision=fp16 \
  --seed="0"

# run: LR1e-5/ rank 32
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-5_rank32" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=32 \
  --mixed_precision=fp16 \
  --seed="0"

# run:1e-6/rank4
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-6_rank4" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=4 \
  --mixed_precision=fp16 \
  --seed="0"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-6_rank8" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=8 \
  --mixed_precision=fp16 \
  --seed="0"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-6_rank16" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=16 \
  --mixed_precision=fp16 \
  --seed="0"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right/lr1e-6_rank32" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right" \
  --instance_prompt="with light coming from sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=1000 \
  --validation_epochs=1 \
  --validation_prompt="photo of a boy wearing a red hat, with light coming from sks" \
  --rank=32 \
  --mixed_precision=fp16 \
  --seed="0"

### FACE 321 lr 1e-4 rank 4
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right_321/lr1e-4_rank4" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right_321" \
  --instance_prompt=", with sunlight illuminate on the sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000000 \
  --validation_epochs=3 \
  --validation_prompt="face of a boy, with sunlight illuminate on the sks" \
  --rank=4 \
  --mixed_precision=fp16 \
  --seed="0"

### FACE 321 lr 1e-4 rank 8
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right_321/lr1e-4_rank8" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right_321" \
  --instance_prompt=", with sunlight illuminate on the sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000000 \
  --validation_epochs=3 \
  --validation_prompt="face of a boy, with sunlight illuminate on the sks" \
  --rank=8 \
  --mixed_precision=fp16 \
  --seed="0"

### FACE 321 lr 1e-4 rank 16
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right_321/lr1e-4_rank16" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right_321" \
  --instance_prompt=", with sunlight illuminate on the sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000000 \
  --validation_epochs=3 \
  --validation_prompt="face of a boy, with sunlight illuminate on the sks" \
  --rank=16 \
  --mixed_precision=fp16 \
  --seed="0"

### FACE 321 lr 1e-4 rank 32
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right_321/lr1e-4_rank32" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right_321" \
  --instance_prompt=", with sunlight illuminate on the sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000000 \
  --validation_epochs=3 \
  --validation_prompt="face of a boy, with sunlight illuminate on the sks" \
  --rank=32 \
  --mixed_precision=fp16 \
  --seed="0"


### FACE 321 lr 1e-5 rank 4
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right_321/lr1e-5_rank4" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right_321" \
  --instance_prompt=", with sunlight illuminate on the sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000000 \
  --validation_epochs=3 \
  --validation_prompt="face of a boy, with sunlight illuminate on the sks" \
  --rank=4 \
  --mixed_precision=fp16 \
  --seed="0"

### FACE 321 lr 1e-5 rank 8
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right_321/lr1e-5_rank8" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right_321" \
  --instance_prompt=", with sunlight illuminate on the sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000000 \
  --validation_epochs=3 \
  --validation_prompt="face of a boy, with sunlight illuminate on the sks" \
  --rank=8 \
  --mixed_precision=fp16 \
  --seed="0"

### FACE 321 lr 1e-5 rank 16
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right_321/lr1e-5_rank16" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right_321" \
  --instance_prompt=", with sunlight illuminate on the sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000000 \
  --validation_epochs=3 \
  --validation_prompt="face of a boy, with sunlight illuminate on the sks" \
  --rank=16 \
  --mixed_precision=fp16 \
  --seed="0"

### FACE 321 lr 1e-5 rank 32
accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"  \
  --output_dir="../../output/20241125_dreambooth_face/right_321/lr1e-5_rank32" \
  --dataset_name="/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right_321" \
  --instance_prompt=", with sunlight illuminate on the sks" \
  --caption_column=caption \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000000 \
  --validation_epochs=3 \
  --validation_prompt="face of a boy, with sunlight illuminate on the sks" \
  --rank=32 \
  --mixed_precision=fp16 \
  --seed="0"

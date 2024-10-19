from pipeline import PureIFPipeline
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline, IFSuperResolutionPipeline
from diffusers.utils import pt_to_pil
#from transformers.utils import FrozenDict

import torch 

MASTER_TYPE = torch.float16
OUTPUT_DIR = "output_ddim_small"

def main():
    print("LOADING PIPELINE...")
    pipe = PureIFPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16, safety_checker=None)    
    print("LOADING SCHEDULER...")
    print(pipe.scheduler.config)
    scheduler_config = {k: v for k, v in pipe.scheduler.config.items() if k != "variance_type"}
    scheduler_config["variance_type"] = "fixed_small"
    pipe.scheduler = DDIMScheduler.from_config(scheduler_config)
    #pipe.enable_model_cpu_offload()
    pipe = pipe.to('cuda')

    # load upscale model 
    super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-M-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
    ).to('cuda')
    #super_res_1_pipe.enable_model_cpu_offload()

    prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
    print("COMPUTING PROMPT EMBEDDINGS...")
    prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
    print("INFERENCING PIPELINE...")
    for seed in range(100):
        image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt", generator=torch.Generator().manual_seed(seed)).images
        pil_image = pt_to_pil(image)
        pil_image[0].save(f"src/20241019/output/{OUTPUT_DIR}/64/seed_{seed:03d}.png")
        image = super_res_1_pipe(
            image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt"
        ).images

        # save intermediate image
        pil_image = pt_to_pil(image)
        pil_image[0].save(f"src/20241019/output/{OUTPUT_DIR}/256/seed_{seed:03d}.png")
        

if __name__ == "__main__":
    main()

    
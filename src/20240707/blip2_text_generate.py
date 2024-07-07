import os 
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import json
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  

scenes = sorted(os.listdir("datasets/standford-orb/llff_crop"))

for scene in tqdm(scenes):
    files = sorted(os.listdir("datasets/standford-orb/llff_crop/" + scene +'/images'))
    outputs = {}
    for filename in tqdm(files):
        if filename.endswith(".jpg"):
            image = Image.open("datasets/standford-orb/llff_crop/" + scene + "/images/" + filename)
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            outputs[filename.replace(".jpg","")] = generated_text
            # do something with the file
    with open("datasets/standford-orb/llff_crop/" + scene + "/prompts.json", "w") as f:
        json.dump(outputs, f, indent=4)
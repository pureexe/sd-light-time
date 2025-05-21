import os
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Settings
base_prefix = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250510_huggingface_controlnet"
scenes = ['val_everett_dining1','val_everett_kitchen2', 'val_everett_kitchen4', 'val_everett_kitchen6']
learning_rates = ['1e-4', '1e-5', '1e-4_normalize32', '1e-5_normalize32']

# Regex to match checkpoint folders
pattern = re.compile(r"^checkpoint-(\d+)$")

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    # Dictionary: checkpoint -> list of psnr values from all scenes
    psnr_by_checkpoint = defaultdict(list)

    for scene in scenes:
        base_dir = os.path.join(base_prefix, scene, lr)
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} does not exist. Skipping scene.")
            continue

        for item in os.listdir(base_dir):
            match = pattern.match(item)
            if match:
                step = int(match.group(1))
                scores_path = os.path.join(base_dir, item, "seed42", "scores.json")
                if os.path.exists(scores_path):
                    try:
                        with open(scores_path, 'r') as f:
                            scores = json.load(f)
                            psnr = scores.get("psnr")
                            if psnr is not None:
                                psnr_by_checkpoint[step].append(psnr)
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse {scores_path}")

    # Compute average PSNR per checkpoint
    if psnr_by_checkpoint:
        sorted_steps = sorted(psnr_by_checkpoint.keys())
        avg_psnr = [sum(psnr_by_checkpoint[step]) / len(psnr_by_checkpoint[step]) for step in sorted_steps]
        plt.plot(sorted_steps, avg_psnr, marker='o', label=f"LR {lr}")
    else:
        print(f"Warning: No PSNR data found for LR {lr}")

plt.xlabel("Checkpoint")
plt.ylabel("Average PSNR across scenes")
plt.title("Average PSNR vs Checkpoint (across scenes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_plot_avg_across_scenes.png")
print("Saved plot as psnr_plot_avg_across_scenes.png")

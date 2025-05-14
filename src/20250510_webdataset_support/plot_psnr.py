import os
import json
import re
import matplotlib.pyplot as plt

# Base directory
base_dir = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250510_huggingface_controlnet/val_everett_kitchen2/1e-4"

# Pattern to match checkpoint directories
pattern = re.compile(r"^checkpoint-(\d+)$")

# Collect psnr values
checkpoints = []
psnr_values = []

for item in os.listdir(base_dir):
    match = pattern.match(item)
    if match:
        step = int(match.group(1))
        scores_path = os.path.join(base_dir, item, "seed42", "scores.json")
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                try:
                    scores = json.load(f)
                    psnr = scores.get("psnr")
                    if psnr is not None:
                        checkpoints.append(step)
                        psnr_values.append(psnr)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse {scores_path}")

# Sort by checkpoint number
sorted_data = sorted(zip(checkpoints, psnr_values))
checkpoints, psnr_values = zip(*sorted_data) if sorted_data else ([], [])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(checkpoints, psnr_values, marker='o')
plt.xlabel("Checkpoint")
plt.ylabel("PSNR")
plt.title("PSNR vs Checkpoint")
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_plot.png")
print("Saved plot as psnr_plot.png")
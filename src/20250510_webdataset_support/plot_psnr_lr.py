import os
import json
import re
import matplotlib.pyplot as plt

# Base directory prefix
base_prefix = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250510_huggingface_controlnet/val_everett_kitchen2"

# Learning rates to include
learning_rates = ['1e-4', '1e-5', '1e-4_normalize32', '1e-5_normalize32']

# Pattern to match checkpoint directories
pattern = re.compile(r"^checkpoint-(\d+)$")

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    base_dir = os.path.join(base_prefix, lr)
    checkpoints = []
    psnr_values = []

    if not os.path.exists(base_dir):
        print(f"Warning: {base_dir} does not exist. Skipping.")
        continue

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

    # Sort and plot if data is available
    if checkpoints:
        sorted_data = sorted(zip(checkpoints, psnr_values))
        checkpoints, psnr_values = zip(*sorted_data)
        plt.plot(checkpoints, psnr_values, marker='o', label=f"LR {lr}")
    else:
        print(f"Warning: No PSNR data found for learning rate {lr}")

plt.xlabel("Checkpoint")
plt.ylabel("PSNR")
plt.title("PSNR vs Checkpoint for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_plot_multiple_lr.png")
print("Saved plot as psnr_plot_multiple_lr.png")
